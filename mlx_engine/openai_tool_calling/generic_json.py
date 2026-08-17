from __future__ import annotations

import copy
import json
from typing import Any

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ParsedToolCalls,
    ToolCallIdFactory,
    build_openai_tool_call,
    tool_names,
)


def build_generic_tool_call_response_schema(
    tool_specs: list[FunctionToolSpec],
    *,
    allow_parallel_tool_calls: bool,
) -> JsonObject:
    tool_call_schemas = [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [tool.name]},
                "arguments": normalize_parameters_schema(tool.parameters),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for tool in tool_specs
    ]
    tool_call_item_schema: JsonObject = (
        tool_call_schemas[0]
        if len(tool_call_schemas) == 1
        else {"oneOf": tool_call_schemas}
    )

    tool_calls_schema: JsonObject = {
        "type": "array",
        "items": tool_call_item_schema,
        "minItems": 1,
    }
    if not allow_parallel_tool_calls or len(tool_specs) == 1:
        tool_calls_schema["maxItems"] = 1

    return {
        "type": "object",
        "properties": {"tool_calls": tool_calls_schema},
        "required": ["tool_calls"],
        "additionalProperties": False,
    }


def normalize_parameters_schema(parameters: JsonObject) -> JsonObject:
    if len(parameters) == 0:
        return {"type": "object", "properties": {}}
    return copy.deepcopy(parameters)


def build_generic_tool_call_instruction(
    tool_specs: list[FunctionToolSpec],
    *,
    allow_parallel_tool_calls: bool,
) -> str:
    tool_definitions = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tool_specs
    ]
    if len(tool_specs) == 1:
        choice_instruction = f"You must call the function named `{tool_specs[0].name}`."
        count_instruction = "Return exactly one tool call."
    elif allow_parallel_tool_calls:
        choice_instruction = "You must call at least one of the available functions."
        count_instruction = (
            "Return one or more tool calls if multiple calls are needed."
        )
    else:
        choice_instruction = "You must call one of the available functions."
        count_instruction = "Return exactly one tool call."

    return "\n".join(
        [
            "Tool calling instructions:",
            choice_instruction,
            count_instruction,
            "Respond only with valid JSON. Do not include prose or markdown.",
            "Use this exact response shape:",
            '{"tool_calls":[{"name":"function_name","arguments":{}}]}',
            "The `arguments` value must be a JSON object matching the selected function's parameters schema.",
            "Available functions:",
            json.dumps(tool_definitions, ensure_ascii=False, indent=2),
        ]
    )


def add_generic_tool_instruction_to_messages(
    messages: list[Any], instruction: str
) -> list[Any]:
    copied_messages = [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]
    if (
        len(copied_messages) > 0
        and isinstance(copied_messages[0], dict)
        and copied_messages[0].get("role") == "system"
        and isinstance(copied_messages[0].get("content"), str)
    ):
        first_message = dict(copied_messages[0])
        first_message["content"] = (
            first_message["content"].rstrip() + "\n\n" + instruction
        )
        copied_messages[0] = first_message
        return copied_messages
    return [{"role": "system", "content": instruction}, *copied_messages]


def parse_generic_tool_call_response(
    model_output: str,
    tool_specs: list[FunctionToolSpec],
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> ParsedToolCalls:
    allowed_tool_names = tool_names(tool_specs)
    if len(allowed_tool_names) == 0:
        return ParsedToolCalls(calls=[], remaining_text=model_output)

    try:
        payload = json.loads(model_output.strip())
    except json.JSONDecodeError:
        return ParsedToolCalls(calls=[], remaining_text=model_output)
    if not isinstance(payload, dict) or set(payload) != {"tool_calls"}:
        return ParsedToolCalls(calls=[], remaining_text=model_output)

    raw_tool_calls = payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list) or len(raw_tool_calls) == 0:
        return ParsedToolCalls(calls=[], remaining_text=model_output)

    calls: list[JsonObject] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict) or set(raw_tool_call) != {
            "name",
            "arguments",
        }:
            return ParsedToolCalls(calls=[], remaining_text=model_output)
        name = raw_tool_call.get("name")
        arguments = raw_tool_call.get("arguments")
        if not isinstance(name, str) or name == "" or not isinstance(arguments, dict):
            return ParsedToolCalls(calls=[], remaining_text=model_output)
        if name not in allowed_tool_names:
            continue
        calls.append(
            build_openai_tool_call(
                name,
                arguments,
                len(calls),
                id_factory=id_factory,
            )
        )

    return ParsedToolCalls(
        calls=calls,
        remaining_text="" if len(calls) > 0 else model_output,
    )
