from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallingValidationError,
    extract_function_tool_specs,
    parse_tool_choice_value,
    validate_strict_tool_calls,
)
from mlx_engine.openai_tool_calling.model_format import parse_model_format_tool_calls


@dataclass(frozen=True)
class ToolCallingPlan:
    tool_specs: list[FunctionToolSpec]

    @property
    def has_active_tools(self) -> bool:
        return len(self.tool_specs) > 0

    @property
    def template_tools(self) -> list[JsonObject]:
        return [tool.to_openai_tool() for tool in self.tool_specs]

    def parse_output(self, model_output: str) -> list[JsonObject]:
        tool_calls = parse_model_format_tool_calls(model_output, self.tool_specs)
        validate_strict_tool_calls(tool_calls, self.tool_specs)
        return tool_calls


def build_tool_calling_plan(
    *,
    tools: list[dict] | None,
    tool_choice_value: Any,
    parallel_tool_calls: bool,
    response_json_schema: str | None,
) -> ToolCallingPlan:
    tool_choice = parse_tool_choice_value(tool_choice_value)
    tool_specs = [] if tool_choice == "none" else extract_function_tool_specs(tools)

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

    return ToolCallingPlan(tool_specs=tool_specs)
