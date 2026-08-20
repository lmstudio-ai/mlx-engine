from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable, Unretrievable
from referencing.jsonschema import DRAFT202012

JsonObject = dict[str, Any]
ToolCallIdFactory = Callable[[], str]
SupportedToolChoice = Literal["none", "auto"]

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DEFAULT_PARAMETERS_SCHEMA: JsonObject = {"type": "object", "properties": {}}
_DEFAULT_STRICT_PARAMETERS_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_FORCED_TOOL_CHOICE_ERROR = (
    "tool_choice='required' and named function tool_choice are not supported yet; "
    "use tool_choice='auto' or tool_choice='none'."
)


class ToolCallingValidationError(ValueError):
    """Raised when OpenAI tool-calling request fields are invalid."""


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
            "parameters": self.parameters,
        }
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}


def parse_tool_choice_value(value: Any) -> SupportedToolChoice | None:
    if value is None or value in ("none", "auto"):
        return value
    if value == "required" or isinstance(value, dict):
        raise ToolCallingValidationError(_FORCED_TOOL_CHOICE_ERROR)
    raise ToolCallingValidationError("tool_choice must be 'none' or 'auto'.")


def extract_function_tool_specs(tools: list[dict] | None) -> list[FunctionToolSpec]:
    if tools is None:
        return []

    specs = [_parse_function_tool(tool, index) for index, tool in enumerate(tools)]
    names = [spec.name for spec in specs]
    duplicate_names = {name for name in names if names.count(name) > 1}
    if duplicate_names:
        raise ToolCallingValidationError(
            f"duplicate function tool name: {sorted(duplicate_names)[0]}"
        )
    return specs


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
                arguments,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
        },
    }


def validate_strict_tool_calls(
    tool_calls: list[JsonObject],
    tool_specs: list[FunctionToolSpec],
) -> None:
    strict_tool_specs = {tool.name: tool for tool in tool_specs if tool.strict}
    for tool_call in tool_calls:
        function = tool_call["function"]
        tool_spec = strict_tool_specs.get(function["name"])
        if tool_spec is None:
            continue
        arguments = _tool_call_arguments_object(tool_spec.name, function)
        try:
            Draft202012Validator(tool_spec.parameters).validate(arguments)
        except ValidationError as error:
            location = f" at {error.json_path}" if error.json_path != "$" else ""
            raise ToolCallingValidationError(
                f"Strict tool call arguments for function `{tool_spec.name}` do not "
                f"match the parameters schema{location}: {error.message}"
            ) from error


def _tool_call_arguments_object(tool_name: str, function: JsonObject) -> JsonObject:
    try:
        arguments = json.loads(function["arguments"])
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


def _parse_function_tool(tool: dict, index: int) -> FunctionToolSpec:
    prefix = f"tools[{index}]"
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
    if _TOOL_NAME_RE.fullmatch(name) is None:
        raise ToolCallingValidationError(
            f"{prefix}.function.name must contain only letters, numbers, "
            "underscores, or dashes and be at most 64 characters"
        )

    if "strict" in function and not isinstance(function["strict"], bool):
        raise ToolCallingValidationError(f"{prefix}.function.strict must be a boolean")
    if "strict" in tool and not isinstance(tool["strict"], bool):
        raise ToolCallingValidationError(f"{prefix}.strict must be a boolean")
    strict = function.get("strict") is True or tool.get("strict") is True
    parameters_missing = "parameters" not in function or function["parameters"] is None
    if parameters_missing:
        parameters = (
            _DEFAULT_STRICT_PARAMETERS_SCHEMA if strict else _DEFAULT_PARAMETERS_SCHEMA
        )
    else:
        parameters = function["parameters"]
    if not isinstance(parameters, dict):
        raise ToolCallingValidationError(
            f"{prefix}.function.parameters must be an object"
        )
    _validate_parameters_schema(name, parameters)

    description = "" if "description" not in function else function["description"]
    if not isinstance(description, str):
        raise ToolCallingValidationError(
            f"{prefix}.function.description must be a string"
        )
    return FunctionToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        strict=strict,
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
    _validate_schema_references(tool_name, parameters)


def _validate_schema_references(tool_name: str, parameters: JsonObject) -> None:
    resolver = (
        Registry()
        .with_resource(
            "",
            DRAFT202012.create_resource(parameters),
        )
        .resolver("")
    )
    for ref in _iter_schema_refs(parameters):
        try:
            resolver.lookup(ref)
        except (NoSuchResource, Unresolvable, Unretrievable) as error:
            raise ToolCallingValidationError(
                f"function tool `{tool_name}` parameters contain an unresolvable "
                f"JSON Schema reference `{ref}`: {error}"
            ) from error


_SCHEMA_VALUE_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)


def _iter_schema_refs(schema: Any):
    if not isinstance(schema, dict):
        return

    ref = schema.get("$ref")
    if isinstance(ref, str):
        yield ref

    # Walk only JSON Schema keyword positions. Keywords like const, enum,
    # default, and examples contain literal instance data where "$ref" is not a
    # schema reference.
    for keyword in _SCHEMA_VALUE_KEYWORDS:
        yield from _iter_schema_refs(schema.get(keyword))

    for keyword in _SCHEMA_ARRAY_KEYWORDS:
        values = schema.get(keyword)
        if isinstance(values, list):
            for value in values:
                yield from _iter_schema_refs(value)

    for keyword in _SCHEMA_MAP_KEYWORDS:
        values = schema.get(keyword)
        if isinstance(values, dict):
            for value in values.values():
                yield from _iter_schema_refs(value)
