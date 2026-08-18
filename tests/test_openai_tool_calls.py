import json
from itertools import count

import pytest

from mlx_engine.openai_tool_calling import (
    ToolCallingValidationError,
    add_generic_tool_instruction_to_messages,
    build_generic_tool_call_instruction,
    build_generic_tool_call_response_schema,
    build_tool_calling_plan,
    extract_function_tool_specs,
    parse_gemma4_arguments_object,
    parse_generic_tool_call_response,
    parse_openai_tool_calls,
    parse_tool_choice_value,
    tool_names,
)
from mlx_engine.tool_protocols import (
    GEMMA4_TOOL_CALL_END,
    GEMMA4_TOOL_CALL_START,
    MUSE_GLIMMER_ATEM_END,
    MUSE_GLIMMER_ATEM_START,
    QWEN35_TOOL_CALL_END,
    QWEN35_TOOL_CALL_START,
)


def _tool(name: str, parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def _specs(*tools):
    return extract_function_tool_specs(list(tools))


def _id_factory():
    counter = count()
    return lambda: f"call_test_{next(counter)}"


def _arguments(call):
    return json.loads(call["function"]["arguments"])


def test_extract_function_tool_specs_requires_openai_function_tools():
    specs = _specs(_tool("lookup"), _tool("search"))

    assert tool_names(specs) == {"lookup", "search"}
    assert specs[0].to_openai_tool() == _tool("lookup")

    with pytest.raises(ToolCallingValidationError, match=r"tools\[0\]\.type"):
        extract_function_tool_specs([{"function": {"name": "legacy_shape"}}])

    with pytest.raises(
        ToolCallingValidationError, match="duplicate function tool name"
    ):
        _specs(_tool("lookup"), _tool("lookup"))


def test_parse_tool_choice_supports_required_and_named_function():
    assert parse_tool_choice_value("required").mode == "required"

    named_choice = parse_tool_choice_value(
        {"type": "function", "function": {"name": "lookup"}}
    )

    assert named_choice.mode == "function"
    assert named_choice.function_name == "lookup"
    assert named_choice.is_forced


def test_generic_tool_call_schema_uses_tool_parameter_schemas():
    weather_parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
        "additionalProperties": False,
    }
    schema = build_generic_tool_call_response_schema(
        _specs(_tool("get_weather", weather_parameters), _tool("search")),
        allow_parallel_tool_calls=False,
    )

    tool_calls_schema = schema["properties"]["tool_calls"]
    assert tool_calls_schema["minItems"] == 1
    assert tool_calls_schema["maxItems"] == 1
    weather_schema = tool_calls_schema["items"]["oneOf"][0]
    assert weather_schema["properties"]["name"] == {
        "type": "string",
        "enum": ["get_weather"],
    }
    assert weather_schema["properties"]["arguments"] == weather_parameters


def test_generic_tool_call_schema_limits_single_available_function():
    schema = build_generic_tool_call_response_schema(
        _specs(_tool("lookup")),
        allow_parallel_tool_calls=True,
    )

    tool_calls_schema = schema["properties"]["tool_calls"]
    assert tool_calls_schema["maxItems"] == 1
    assert tool_calls_schema["items"]["properties"]["name"]["enum"] == ["lookup"]


def test_generic_tool_instruction_is_merged_into_existing_system_message():
    instruction = build_generic_tool_call_instruction(
        _specs(_tool("lookup")),
        allow_parallel_tool_calls=True,
    )

    messages = add_generic_tool_instruction_to_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Find Paris."},
        ],
        instruction,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Be concise." in messages[0]["content"]
    assert "Tool calling instructions:" in messages[0]["content"]
    assert "lookup" in messages[0]["content"]


def test_generic_json_response_parses_as_openai_tool_calls():
    output = json.dumps(
        {
            "tool_calls": [
                {"name": "lookup", "arguments": {"query": "weather"}},
                {"name": "search", "arguments": {"query": "Paris"}},
            ]
        }
    )

    result = parse_generic_tool_call_response(
        output,
        _specs(_tool("lookup"), _tool("search")),
        id_factory=_id_factory(),
    )

    assert [call["id"] for call in result.calls] == ["call_test_0", "call_test_1"]
    assert [call["index"] for call in result.calls] == [0, 1]
    assert [call["function"]["name"] for call in result.calls] == [
        "lookup",
        "search",
    ]
    assert _arguments(result.calls[0]) == {"query": "weather"}
    assert _arguments(result.calls[1]) == {"query": "Paris"}
    assert result.remaining_text == ""


def test_generic_json_response_filters_unavailable_tools():
    output = json.dumps(
        {
            "tool_calls": [
                {"name": "search", "arguments": {"query": "wrong"}},
                {"name": "lookup", "arguments": {"query": "right"}},
            ]
        }
    )

    result = parse_generic_tool_call_response(
        output,
        _specs(_tool("lookup")),
        id_factory=_id_factory(),
    )

    assert len(result.calls) == 1
    assert result.calls[0]["function"]["name"] == "lookup"
    assert _arguments(result.calls[0]) == {"query": "right"}


def test_generic_json_response_rejects_non_canonical_arguments_string():
    output = json.dumps(
        {"tool_calls": [{"name": "lookup", "arguments": '{"query":"right"}'}]}
    )

    result = parse_generic_tool_call_response(
        output,
        _specs(_tool("lookup")),
        id_factory=_id_factory(),
    )

    assert result.calls == []
    assert result.remaining_text == output


def test_tool_calling_plan_accepts_valid_tool_parameter_schema():
    plan = build_tool_calling_plan(
        messages=[{"role": "user", "content": "Find Paris."}],
        tools=[
            _tool(
                "lookup",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ],
        tool_choice_value="auto",
        parallel_tool_calls=True,
        response_json_schema=None,
    )

    assert plan.strategy == "native"


def test_tool_calling_plan_rejects_invalid_tool_parameter_schema():
    with pytest.raises(ToolCallingValidationError) as error:
        build_tool_calling_plan(
            messages=[{"role": "user", "content": "Find Paris."}],
            tools=[
                _tool(
                    "lookup",
                    {
                        "type": "object",
                        "properties": {"query": {"type": "not-a-json-schema-type"}},
                    },
                )
            ],
            tool_choice_value="required",
            parallel_tool_calls=True,
            response_json_schema=None,
        )

    message = str(error.value)
    assert "lookup" in message
    assert "parameters" in message
    assert "not-a-json-schema-type" in message


def test_tool_calling_plan_rejects_invalid_tool_parameter_schema_for_auto_choice():
    with pytest.raises(ToolCallingValidationError):
        build_tool_calling_plan(
            messages=[{"role": "user", "content": "Find Paris."}],
            tools=[
                _tool(
                    "lookup",
                    {"type": "object", "required": "query"},
                )
            ],
            tool_choice_value="auto",
            parallel_tool_calls=True,
            response_json_schema=None,
        )


def test_tool_calling_plan_uses_generic_json_for_required_choice():
    plan = build_tool_calling_plan(
        messages=[{"role": "user", "content": "Find Paris."}],
        tools=[_tool("lookup")],
        tool_choice_value="required",
        parallel_tool_calls=True,
        response_json_schema=None,
    )

    assert plan.strategy == "generic_json"
    assert plan.template_tools is None
    assert plan.template_tool_choice is None
    assert plan.should_buffer_output
    assert plan.requires_tool_call
    assert plan.generation_json_schema is not None
    assert "Tool calling instructions:" in plan.prompt_messages[0]["content"]

    parsed = plan.parse_output(
        json.dumps({"tool_calls": [{"name": "lookup", "arguments": {}}]})
    )
    assert len(parsed.calls) == 1
    assert parsed.calls[0]["function"]["name"] == "lookup"


def test_tool_calling_plan_keeps_native_path_for_auto_choice():
    tools = [_tool("lookup")]
    response_schema = '{"type":"object"}'

    plan = build_tool_calling_plan(
        messages=[{"role": "user", "content": "Find Paris."}],
        tools=tools,
        tool_choice_value="auto",
        parallel_tool_calls=True,
        response_json_schema=response_schema,
    )

    assert plan.strategy == "native"
    assert plan.template_tools == tools
    assert plan.template_tool_choice == "auto"
    assert plan.should_buffer_output
    assert plan.max_tool_calls == 1
    assert plan.generation_json_schema == response_schema


def test_tool_calling_plan_returns_only_one_tool_call_for_serial_mvp():
    plan = build_tool_calling_plan(
        messages=[{"role": "user", "content": "Find Paris."}],
        tools=[_tool("lookup"), _tool("search")],
        tool_choice_value="auto",
        parallel_tool_calls=True,
        response_json_schema=None,
    )
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
        f"{QWEN35_TOOL_CALL_START}<function=search></function>{QWEN35_TOOL_CALL_END}"
    )

    result = plan.parse_output(output)

    assert [call["function"]["name"] for call in result.calls] == ["lookup"]
    assert result.remaining_text == ""


def test_qwen35_tool_call_parses_parameters_as_openai_tool_call():
    output = f"""Before
{QWEN35_TOOL_CALL_START}
<function=lookup>
<parameter=query>{{"city":"Paris"}}</parameter>
<parameter=count>2</parameter>
<parameter=label> plain text </parameter>
</function>
{QWEN35_TOOL_CALL_END}
After"""

    result = parse_openai_tool_calls(
        output, _specs(_tool("lookup")), id_factory=_id_factory()
    )

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call["index"] == 0
    assert call["id"] == "call_test_0"
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup"
    assert _arguments(call) == {
        "query": {"city": "Paris"},
        "count": 2,
        "label": "plain text",
    }
    assert QWEN35_TOOL_CALL_START not in result.remaining_text
    assert QWEN35_TOOL_CALL_END not in result.remaining_text
    assert "Before" in result.remaining_text
    assert "After" in result.remaining_text


def test_qwen35_tool_call_parses_multiple_calls_in_emission_order():
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
        f"{QWEN35_TOOL_CALL_START}<function=search-tool>"
        "<parameter=query>weather</parameter></function>"
        f"{QWEN35_TOOL_CALL_END}"
    )

    result = parse_openai_tool_calls(
        output,
        _specs(_tool("lookup"), _tool("search-tool")),
        id_factory=_id_factory(),
    )

    assert [call["index"] for call in result.calls] == [0, 1]
    assert [call["id"] for call in result.calls] == ["call_test_0", "call_test_1"]
    assert [call["function"]["name"] for call in result.calls] == [
        "lookup",
        "search-tool",
    ]
    assert _arguments(result.calls[0]) == {}
    assert _arguments(result.calls[1]) == {"query": "weather"}


def test_gemma4_argument_parser_supports_native_object_syntax():
    parsed = parse_gemma4_arguments_object(
        '{city:<|"|>Paris<|"|>,count:2,score:-1.5e2,enabled:true,'
        'missing:None,nested:{unit:<|"|>celsius<|"|>},items:[1,none,false]}'
    )

    assert parsed == {
        "city": "Paris",
        "count": 2,
        "score": -150.0,
        "enabled": True,
        "missing": None,
        "nested": {"unit": "celsius"},
        "items": [1, None, False],
    }


def test_gemma4_tool_call_parses_native_call_as_openai_tool_call():
    output = (
        "prefix "
        f"{GEMMA4_TOOL_CALL_START}"
        'call:mcp:lookup{city:<|"|>Paris<|"|>,metadata:{source:<|"|>wx<|"|>}}'
        f"{GEMMA4_TOOL_CALL_END}"
        " suffix"
    )

    result = parse_openai_tool_calls(
        output, _specs(_tool("mcp:lookup")), id_factory=_id_factory()
    )

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call["index"] == 0
    assert call["id"] == "call_test_0"
    assert call["function"]["name"] == "mcp:lookup"
    assert _arguments(call) == {"city": "Paris", "metadata": {"source": "wx"}}
    assert GEMMA4_TOOL_CALL_START not in result.remaining_text
    assert GEMMA4_TOOL_CALL_END not in result.remaining_text
    assert "prefix" in result.remaining_text
    assert "suffix" in result.remaining_text


def test_muse_glimmer_tool_call_parses_atem_invocation_as_openai_tool_call():
    output = (
        "prefix "
        f"{MUSE_GLIMMER_ATEM_START}"
        '<atem:invoke name="lookup">'
        '<atem:parameter name="query">weather</atem:parameter>'
        '<atem:parameter name="metadata">{"city":"Paris"}</atem:parameter>'
        "</atem:invoke>"
        f"{MUSE_GLIMMER_ATEM_END}"
        " suffix"
    )

    result = parse_openai_tool_calls(
        output, _specs(_tool("lookup")), id_factory=_id_factory()
    )

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call["function"]["name"] == "lookup"
    assert _arguments(call) == {"query": "weather", "metadata": {"city": "Paris"}}
    assert MUSE_GLIMMER_ATEM_START not in result.remaining_text
    assert MUSE_GLIMMER_ATEM_END not in result.remaining_text
    assert "prefix" in result.remaining_text
    assert "suffix" in result.remaining_text


def test_parser_removes_native_blocks_for_unknown_tool_names():
    output = (
        "prefix "
        f"{QWEN35_TOOL_CALL_START}<function=unknown></function>{QWEN35_TOOL_CALL_END}"
        f"{GEMMA4_TOOL_CALL_START}call:also_unknown{{}}{GEMMA4_TOOL_CALL_END}"
        f"{MUSE_GLIMMER_ATEM_START}"
        '<atem:invoke name="muse_unknown"></atem:invoke>'
        f"{MUSE_GLIMMER_ATEM_END}"
        " suffix"
    )

    result = parse_openai_tool_calls(
        output, _specs(_tool("allowed")), id_factory=_id_factory()
    )

    assert result.calls == []
    assert QWEN35_TOOL_CALL_START not in result.remaining_text
    assert GEMMA4_TOOL_CALL_START not in result.remaining_text
    assert MUSE_GLIMMER_ATEM_START not in result.remaining_text
    assert "prefix" in result.remaining_text
    assert "suffix" in result.remaining_text


def test_parser_leaves_output_unchanged_without_tool_specs():
    output = f"visible {QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"

    result = parse_openai_tool_calls(output, [], id_factory=_id_factory())

    assert result.calls == []
    assert result.remaining_text == output
