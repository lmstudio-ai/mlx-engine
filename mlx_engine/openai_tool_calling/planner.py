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
from mlx_engine.openai_tool_calling.model_format import (
    ModelToolCallFormat,
    parse_model_format_tool_calls,
)


@dataclass(frozen=True)
class ToolCallingPlan:
    tool_specs: list[FunctionToolSpec]
    model_format: ModelToolCallFormat = "auto"

    @property
    def has_active_tools(self) -> bool:
        return len(self.tool_specs) > 0

    @property
    def template_tools(self) -> list[JsonObject]:
        return [tool.to_openai_tool() for tool in self.tool_specs]

    def parse_output(self, model_output: str) -> list[JsonObject]:
        tool_calls = parse_model_format_tool_calls(
            model_output,
            self.tool_specs,
            model_format=self.model_format,
        )
        validate_strict_tool_calls(tool_calls, self.tool_specs)
        return tool_calls


def build_tool_calling_plan(
    *,
    tools: list[dict] | None,
    tool_choice_value: Any,
    parallel_tool_calls: bool,
    response_json_schema: str | None,
    model_type: str | None = None,
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

    return ToolCallingPlan(
        tool_specs=tool_specs,
        model_format=_model_format_from_model_type(model_type),
    )


def _model_format_from_model_type(model_type: str | None) -> ModelToolCallFormat:
    normalized_model_type = str(model_type or "")
    if normalized_model_type.startswith("qwen3_5"):
        return "qwen35"
    if normalized_model_type.startswith("gemma4"):
        return "gemma4"
    if normalized_model_type == "muse_glimmer":
        return "muse_glimmer"
    return "auto"
