from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    ParsedToolCalls,
    ToolCallingValidationError,
    extract_function_tool_specs,
    parse_tool_choice_value,
    validate_strict_tool_calls,
)
from mlx_engine.openai_tool_calling.native import parse_native_tool_calls

ToolCallingStrategy = Literal["none", "model_format"]


@dataclass(frozen=True)
class ToolCallingPlan:
    strategy: ToolCallingStrategy
    tool_specs: list[FunctionToolSpec]
    prompt_messages: list[Any]
    template_tools: list[Any] | None
    template_tool_choice: Any
    generation_json_schema: str | None
    max_tool_calls: int = 1

    @property
    def has_active_tools(self) -> bool:
        return len(self.tool_specs) > 0

    def parse_output(self, model_output: str) -> ParsedToolCalls:
        if self.strategy == "model_format":
            parsed = parse_native_tool_calls(model_output, self.tool_specs)
            validate_strict_tool_calls(parsed, self.tool_specs)
            return parsed
        return ParsedToolCalls(calls=[], remaining_text=model_output)


def build_tool_calling_plan(
    *,
    messages: list[Any],
    tools: list[Any] | None,
    tool_choice_value: Any,
    parallel_tool_calls: bool,
    response_json_schema: str | None,
) -> ToolCallingPlan:
    tool_choice = parse_tool_choice_value(tool_choice_value)
    tool_specs = extract_function_tool_specs(tools)

    if tool_choice is not None and tool_choice.mode == "none":
        tool_specs = []

    if len(tool_specs) > 0 and response_json_schema is not None:
        raise ToolCallingValidationError(
            "response_format is not supported with active tools; "
            "set tool_choice='none' or omit response_format."
        )

    if parallel_tool_calls and len(tool_specs) > 0:
        raise ToolCallingValidationError(
            "parallel_tool_calls=true is not supported with tools; "
            "set parallel_tool_calls=false."
        )

    if tool_choice is not None and tool_choice.is_forced:
        raise ToolCallingValidationError(
            "tool_choice='required' and named function tool_choice are not "
            "supported yet; use tool_choice='auto' or tool_choice='none'."
        )

    return ToolCallingPlan(
        strategy="model_format" if len(tool_specs) > 0 else "none",
        tool_specs=tool_specs,
        prompt_messages=messages,
        template_tools=[tool.to_openai_tool() for tool in tool_specs]
        if len(tool_specs) > 0
        else None,
        template_tool_choice="auto"
        if tool_choice is not None
        and tool_choice.mode == "auto"
        and len(tool_specs) > 0
        else None,
        generation_json_schema=response_json_schema,
        max_tool_calls=1,
    )
