from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from mlx_engine.openai_tool_calling.generic_json import (
    add_generic_tool_instruction_to_messages,
    build_generic_tool_call_instruction,
    build_generic_tool_call_response_schema,
    parse_generic_tool_call_response,
)
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    OpenAIToolChoice,
    ParsedToolCalls,
    ToolCallingValidationError,
    extract_function_tool_specs,
    parse_tool_choice_value,
)
from mlx_engine.openai_tool_calling.native import parse_native_tool_calls

ToolCallingStrategy = Literal["none", "native", "generic_json"]


@dataclass(frozen=True)
class ToolCallingPlan:
    strategy: ToolCallingStrategy
    tool_specs: list[FunctionToolSpec]
    prompt_messages: list[Any]
    template_tools: list[Any] | None
    template_tool_choice: Any
    generation_json_schema: str | None

    @property
    def has_active_tools(self) -> bool:
        return len(self.tool_specs) > 0

    @property
    def should_buffer_output(self) -> bool:
        return self.has_active_tools

    @property
    def requires_tool_call(self) -> bool:
        return self.strategy == "generic_json"

    def parse_output(self, model_output: str) -> ParsedToolCalls:
        if self.strategy == "generic_json":
            return parse_generic_tool_call_response(model_output, self.tool_specs)
        if self.strategy == "native":
            return parse_native_tool_calls(model_output, self.tool_specs)
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

    if tool_choice is not None and tool_choice.is_forced:
        selected_tool_specs = _select_forced_tool_specs(tool_specs, tool_choice)
        allow_parallel_tool_calls = (
            parallel_tool_calls
            and tool_choice.mode != "function"
            and len(tool_specs) > 1
        )
        instruction = build_generic_tool_call_instruction(
            selected_tool_specs,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )
        generation_json_schema = json.dumps(
            build_generic_tool_call_response_schema(
                selected_tool_specs,
                allow_parallel_tool_calls=allow_parallel_tool_calls,
            )
        )
        return ToolCallingPlan(
            strategy="generic_json",
            tool_specs=selected_tool_specs,
            prompt_messages=add_generic_tool_instruction_to_messages(
                messages,
                instruction,
            ),
            template_tools=None,
            template_tool_choice=None,
            generation_json_schema=generation_json_schema,
        )

    return ToolCallingPlan(
        strategy="native" if len(tool_specs) > 0 else "none",
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
    )


def _select_forced_tool_specs(
    tool_specs: list[FunctionToolSpec], tool_choice: OpenAIToolChoice
) -> list[FunctionToolSpec]:
    if len(tool_specs) == 0:
        raise ToolCallingValidationError(
            "forced tool_choice requires at least one function tool"
        )
    if tool_choice.mode != "function":
        return tool_specs

    selected_tool_specs = [
        tool for tool in tool_specs if tool.name == tool_choice.function_name
    ]
    if len(selected_tool_specs) == 0:
        raise ToolCallingValidationError(
            f"tool_choice requested unknown function: {tool_choice.function_name}"
        )
    return selected_tool_specs
