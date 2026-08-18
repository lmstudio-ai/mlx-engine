from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    OpenAIToolChoice,
    ParsedToolCalls,
    ToolCallingValidationError,
    ToolCallIdFactory,
    build_openai_tool_call,
    extract_function_tool_specs,
    parse_tool_choice_value,
    tool_names,
)
from mlx_engine.openai_tool_calling.native import (
    parse_gemma4_arguments_object,
    parse_gemma4_tool_calls,
    parse_muse_glimmer_tool_calls,
    parse_native_tool_calls,
    parse_qwen35_tool_argument_value,
    parse_qwen35_tool_calls,
    remove_native_tool_call_blocks,
)
from mlx_engine.openai_tool_calling.planner import (
    ToolCallingPlan,
    build_tool_calling_plan,
)

# Compatibility alias for the initial endpoint implementation.
parse_openai_tool_calls = parse_native_tool_calls

__all__ = [
    "FunctionToolSpec",
    "JsonObject",
    "OpenAIToolChoice",
    "ParsedToolCalls",
    "ToolCallingPlan",
    "ToolCallingValidationError",
    "ToolCallIdFactory",
    "build_openai_tool_call",
    "build_tool_calling_plan",
    "extract_function_tool_specs",
    "parse_gemma4_arguments_object",
    "parse_gemma4_tool_calls",
    "parse_muse_glimmer_tool_calls",
    "parse_native_tool_calls",
    "parse_openai_tool_calls",
    "parse_qwen35_tool_argument_value",
    "parse_qwen35_tool_calls",
    "parse_tool_choice_value",
    "remove_native_tool_call_blocks",
    "tool_names",
]
