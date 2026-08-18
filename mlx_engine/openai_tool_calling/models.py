from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from jsonschema import Draft202012Validator, SchemaError, ValidationError

JsonObject = dict[str, Any]
ToolCallIdFactory = Callable[[], str]
ToolChoiceMode = Literal["none", "auto", "required", "function"]

_DEFAULT_PARAMETERS_SCHEMA: JsonObject = {"type": "object", "properties": {}}


class ToolCallingValidationError(ValueError):
    """Raised when OpenAI tool-calling request fields are invalid."""


@dataclass(frozen=True)
class OpenAIToolChoice:
    mode: ToolChoiceMode
    function_name: str | None = None

    @property
    def is_forced(self) -> bool:
        return self.mode in ("required", "function")


@dataclass(frozen=True)
class FunctionToolSpec:
    name: str
    description: str
    parameters: JsonObject
    strict: bool

    def to_openai_tool(self) -> JsonObject:
        function: JsonObject = {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}


@dataclass(frozen=True)
class ParsedToolCalls:
    calls: list[JsonObject]
    remaining_text: str


def parse_tool_choice_value(value: Any) -> OpenAIToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in ("none", "auto", "required"):
            return OpenAIToolChoice(mode=value)
        raise ToolCallingValidationError(
            "tool_choice must be one of: none, auto, required"
        )
    if isinstance(value, dict):
        if value.get("type") != "function":
            raise ToolCallingValidationError("tool_choice.type must be 'function'")
        function = value.get("function")
        if not isinstance(function, dict):
            raise ToolCallingValidationError("tool_choice.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or name == "":
            raise ToolCallingValidationError(
                "tool_choice.function.name must be a non-empty string"
            )
        return OpenAIToolChoice(mode="function", function_name=name)
    raise ToolCallingValidationError(
        "tool_choice must be a string or function choice object"
    )


def extract_function_tool_specs(tools: list[Any] | None) -> list[FunctionToolSpec]:
    if tools is None:
        return []

    specs = [_parse_function_tool(tool, index) for index, tool in enumerate(tools)]
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ToolCallingValidationError(
                f"duplicate function tool name: {spec.name}"
            )
        seen_names.add(spec.name)
    return specs


def tool_names(tool_specs: list[FunctionToolSpec]) -> set[str]:
    return {tool.name for tool in tool_specs}


def build_openai_tool_call(
    tool_name: str,
    arguments: JsonObject,
    index: int,
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> JsonObject:
    return {
        "type": "function",
        "index": index,
        "id": (id_factory or default_tool_call_id)(),
        "function": {
            "name": tool_name,
            "arguments": json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            ),
        },
    }


def validate_strict_tool_calls(
    parsed_tool_calls: ParsedToolCalls,
    tool_specs: list[FunctionToolSpec],
) -> None:
    strict_tool_specs = {tool.name: tool for tool in tool_specs if tool.strict}
    if len(strict_tool_specs) == 0:
        return

    for tool_call in parsed_tool_calls.calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = function.get("name")
        if not isinstance(tool_name, str):
            continue
        tool_spec = strict_tool_specs.get(tool_name)
        if tool_spec is None:
            continue
        arguments = _tool_call_arguments_object(tool_name, function)
        try:
            Draft202012Validator(tool_spec.parameters).validate(arguments)
        except ValidationError as error:
            location = f" at {error.json_path}" if error.json_path != "$" else ""
            raise ToolCallingValidationError(
                f"Strict tool call arguments for function `{tool_name}` do not "
                f"match the parameters schema{location}: {error.message}"
            ) from error


def _tool_call_arguments_object(tool_name: str, function: JsonObject) -> JsonObject:
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise ToolCallingValidationError(
            f"Strict tool call arguments for function `{tool_name}` must be a JSON object."
        )
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ToolCallingValidationError(
            f"Strict tool call arguments for function `{tool_name}` must be valid JSON."
        ) from error
    if not isinstance(arguments, dict):
        raise ToolCallingValidationError(
            f"Strict tool call arguments for function `{tool_name}` must be a JSON object."
        )
    return arguments


def default_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def _parse_function_tool(tool: Any, index: int) -> FunctionToolSpec:
    prefix = f"tools[{index}]"
    if not isinstance(tool, dict):
        raise ToolCallingValidationError(f"{prefix} must be an object")
    if tool.get("type") != "function":
        raise ToolCallingValidationError(f"{prefix}.type must be 'function'")

    function = tool.get("function")
    if not isinstance(function, dict):
        raise ToolCallingValidationError(f"{prefix}.function must be an object")

    name = function.get("name")
    if not isinstance(name, str) or name == "":
        raise ToolCallingValidationError(
            f"{prefix}.function.name must be a non-empty string"
        )

    parameters = function.get("parameters", _DEFAULT_PARAMETERS_SCHEMA)
    if parameters is None:
        parameters = _DEFAULT_PARAMETERS_SCHEMA
    if not isinstance(parameters, dict):
        raise ToolCallingValidationError(
            f"{prefix}.function.parameters must be an object"
        )
    parameters = copy.deepcopy(parameters)
    _validate_parameters_schema(name, parameters)

    description = function.get("description")
    return FunctionToolSpec(
        name=name,
        description=description if isinstance(description, str) else "",
        parameters=parameters,
        strict=function.get("strict") is True or tool.get("strict") is True,
    )


def _validate_parameters_schema(tool_name: str, parameters: JsonObject) -> None:
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as error:
        location = f" at {error.json_path}" if error.json_path != "$" else ""
        raise ToolCallingValidationError(
            f"function tool `{tool_name}` parameters must be a valid JSON Schema"
            f"{location}: {error.message}"
        ) from error
