import json
from itertools import count

import pytest

from mlx_engine.openai_tool_calling import (
    ToolCallingValidationError,
    build_tool_calling_plan,
    extract_function_tool_specs,
    parse_gemma4_arguments_object,
    parse_model_format_tool_calls,
    parse_tool_choice_value,
)
from mlx_engine.tool_protocols import (
    GEMMA4_TOOL_CALL_END,
    GEMMA4_TOOL_CALL_START,
    MUSE_GLIMMER_ATEM_END,
    MUSE_GLIMMER_ATEM_START,
    QWEN35_TOOL_CALL_END,
    QWEN35_TOOL_CALL_START,
)


_MISSING = object()


def _tool(name: str, parameters=_MISSING, *, strict: bool = False):
    function = {
        "name": name,
        "description": f"Call {name}",
        "parameters": {"type": "object", "properties": {}}
        if parameters is _MISSING
        else parameters,
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _specs(*tools):
    return extract_function_tool_specs(list(tools))


def _id_factory():
    counter = count()
    return lambda: f"call_test_{next(counter)}"


def _arguments(call):
    return json.loads(call["function"]["arguments"])


def _plan(*tools):
    return build_tool_calling_plan(
        tools=list(tools),
        tool_choice_value="auto",
        parallel_tool_calls=False,
        response_json_schema=None,
    )


def test_extract_function_tool_specs_requires_openai_function_tools():
    specs = _specs(_tool("lookup"), _tool("search"))

    assert {spec.name for spec in specs} == {"lookup", "search"}
    assert specs[0].to_openai_tool() == _tool("lookup")

    with pytest.raises(ToolCallingValidationError, match=r"tools\[0\]\.type"):
        extract_function_tool_specs([{"function": {"name": "legacy_shape"}}])

    with pytest.raises(
        ToolCallingValidationError, match="duplicate function tool name"
    ):
        _specs(_tool("lookup"), _tool("lookup"))


@pytest.mark.parametrize("parameters", [False, [], "", 0])
def test_extract_function_tool_specs_rejects_falsy_non_object_parameters(parameters):
    with pytest.raises(
        ToolCallingValidationError, match="parameters must be an object"
    ):
        _specs(_tool("lookup", parameters))


def test_extract_function_tool_specs_preserves_empty_schema():
    specs = _specs(_tool("lookup", {}))

    assert specs[0].parameters == {}
    assert specs[0].to_openai_tool()["function"]["parameters"] == {}


def test_parse_tool_choice_supports_only_auto_and_none_for_mvp():
    assert parse_tool_choice_value(None) is None
    assert parse_tool_choice_value("auto") == "auto"
    assert parse_tool_choice_value("none") == "none"

    with pytest.raises(ToolCallingValidationError, match="not supported"):
        parse_tool_choice_value("required")
    with pytest.raises(ToolCallingValidationError, match="not supported"):
        parse_tool_choice_value({"type": "function", "function": {"name": "lookup"}})
    with pytest.raises(ToolCallingValidationError, match="tool_choice"):
        parse_tool_choice_value("invalid")


def test_tool_calling_plan_accepts_valid_tool_parameter_schema():
    plan = build_tool_calling_plan(
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
        parallel_tool_calls=False,
        response_json_schema=None,
    )

    assert plan.has_active_tools


def test_tool_calling_plan_rejects_parallel_tool_calls_with_active_tools():
    with pytest.raises(ToolCallingValidationError, match="parallel_tool_calls"):
        build_tool_calling_plan(
            tools=[_tool("lookup")],
            tool_choice_value="auto",
            parallel_tool_calls=True,
            response_json_schema=None,
        )


def test_tool_calling_plan_allows_parallel_flag_when_tool_choice_is_none():
    plan = build_tool_calling_plan(
        tools=[_tool("lookup")],
        tool_choice_value="none",
        parallel_tool_calls=True,
        response_json_schema=None,
    )

    assert not plan.has_active_tools


def test_tool_calling_plan_rejects_response_format_with_active_tools():
    with pytest.raises(ToolCallingValidationError, match="response_format"):
        build_tool_calling_plan(
            tools=[_tool("lookup")],
            tool_choice_value="auto",
            parallel_tool_calls=False,
            response_json_schema='{"type":"object"}',
        )


def test_tool_calling_plan_allows_response_format_when_tool_choice_is_none():
    plan = build_tool_calling_plan(
        tools=[_tool("lookup")],
        tool_choice_value="none",
        parallel_tool_calls=False,
        response_json_schema='{"type":"object"}',
    )

    assert not plan.has_active_tools


def test_tool_calling_plan_rejects_invalid_tool_parameter_schema():
    with pytest.raises(ToolCallingValidationError) as error:
        build_tool_calling_plan(
            tools=[
                _tool(
                    "lookup",
                    {
                        "type": "object",
                        "properties": {"query": {"type": "not-a-json-schema-type"}},
                    },
                )
            ],
            tool_choice_value="auto",
            parallel_tool_calls=False,
            response_json_schema=None,
        )

    message = str(error.value)
    assert "lookup" in message
    assert "parameters" in message
    assert "not-a-json-schema-type" in message


@pytest.mark.parametrize(
    "tool_choice_value",
    [
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ],
)
def test_tool_calling_plan_rejects_forced_tool_choice(tool_choice_value):
    with pytest.raises(ToolCallingValidationError, match="not supported"):
        build_tool_calling_plan(
            tools=[_tool("lookup")],
            tool_choice_value=tool_choice_value,
            parallel_tool_calls=False,
            response_json_schema=None,
        )


def test_tool_calling_plan_uses_model_format_parser_for_auto_choice():
    tools = [_tool("lookup")]

    plan = build_tool_calling_plan(
        tools=tools,
        tool_choice_value="auto",
        parallel_tool_calls=False,
        response_json_schema=None,
    )

    assert plan.has_active_tools
    assert plan.template_tools == tools


def test_tool_calling_plan_validates_strict_tool_arguments():
    plan = _plan(
        _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
            strict=True,
        )
    )
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup>"
        "<parameter=count>2</parameter></function>"
        f"{QWEN35_TOOL_CALL_END}"
    )

    result = plan.parse_output(output)

    assert len(result) == 1
    assert _arguments(result[0]) == {"count": 2}


def test_tool_calling_plan_rejects_invalid_strict_tool_arguments():
    plan = _plan(
        _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
            strict=True,
        )
    )
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup>"
        "<parameter=count>not an integer</parameter></function>"
        f"{QWEN35_TOOL_CALL_END}"
    )

    with pytest.raises(
        ToolCallingValidationError,
        match="Strict tool call arguments.*count.*integer",
    ):
        plan.parse_output(output)


def test_tool_calling_plan_allows_invalid_non_strict_tool_arguments():
    plan = _plan(
        _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        )
    )
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup>"
        "<parameter=count>not an integer</parameter></function>"
        f"{QWEN35_TOOL_CALL_END}"
    )

    result = plan.parse_output(output)

    assert len(result) == 1
    assert _arguments(result[0]) == {"count": "not an integer"}


def test_tool_calling_plan_preserves_all_parsed_tool_calls():
    plan = _plan(_tool("lookup"), _tool("search"))
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
        f"{QWEN35_TOOL_CALL_START}<function=search></function>{QWEN35_TOOL_CALL_END}"
    )

    result = plan.parse_output(output)

    assert [call["function"]["name"] for call in result] == [
        "lookup",
        "search",
    ]


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

    result = parse_model_format_tool_calls(
        output, _specs(_tool("lookup")), id_factory=_id_factory()
    )

    assert len(result) == 1
    call = result[0]
    assert call["index"] == 0
    assert call["id"] == "call_test_0"
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup"
    assert _arguments(call) == {
        "query": {"city": "Paris"},
        "count": 2,
        "label": "plain text",
    }


def test_qwen35_tool_call_parses_multiple_calls_in_emission_order():
    output = (
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
        f"{QWEN35_TOOL_CALL_START}<function=search-tool>"
        "<parameter=query>weather</parameter></function>"
        f"{QWEN35_TOOL_CALL_END}"
    )

    result = parse_model_format_tool_calls(
        output,
        _specs(_tool("lookup"), _tool("search-tool")),
        id_factory=_id_factory(),
    )

    assert [call["index"] for call in result] == [0, 1]
    assert [call["id"] for call in result] == ["call_test_0", "call_test_1"]
    assert [call["function"]["name"] for call in result] == [
        "lookup",
        "search-tool",
    ]
    assert _arguments(result[0]) == {}
    assert _arguments(result[1]) == {"query": "weather"}


def test_parser_rejects_mixed_model_format_tool_calls():
    output = (
        f'{GEMMA4_TOOL_CALL_START}call:search{{query:<|"|>news<|"|>}}'
        f"{GEMMA4_TOOL_CALL_END}"
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
    )

    with pytest.raises(ToolCallingValidationError, match="Mixed model-format"):
        parse_model_format_tool_calls(
            output,
            _specs(_tool("lookup"), _tool("search")),
            id_factory=_id_factory(),
        )


def test_gemma4_argument_parser_supports_model_format_object_syntax():
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


def test_gemma4_tool_call_parses_model_format_call_as_openai_tool_call():
    output = (
        "prefix "
        f"{GEMMA4_TOOL_CALL_START}"
        'call:mcp:lookup{city:<|"|>Paris<|"|>,metadata:{source:<|"|>wx<|"|>}}'
        f"{GEMMA4_TOOL_CALL_END}"
        " suffix"
    )

    result = parse_model_format_tool_calls(
        output, _specs(_tool("mcp:lookup")), id_factory=_id_factory()
    )

    assert len(result) == 1
    call = result[0]
    assert call["index"] == 0
    assert call["id"] == "call_test_0"
    assert call["function"]["name"] == "mcp:lookup"
    assert _arguments(call) == {"city": "Paris", "metadata": {"source": "wx"}}


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

    result = parse_model_format_tool_calls(
        output, _specs(_tool("lookup")), id_factory=_id_factory()
    )

    assert len(result) == 1
    call = result[0]
    assert call["function"]["name"] == "lookup"
    assert _arguments(call) == {"query": "weather", "metadata": {"city": "Paris"}}


def test_parser_ignores_unknown_model_format_blocks():
    output = (
        "prefix "
        f"{QWEN35_TOOL_CALL_START}<function=unknown></function>{QWEN35_TOOL_CALL_END}"
        f"{GEMMA4_TOOL_CALL_START}call:also_unknown{{}}{GEMMA4_TOOL_CALL_END}"
        f"{MUSE_GLIMMER_ATEM_START}"
        '<atem:invoke name="muse_unknown"></atem:invoke>'
        f"{MUSE_GLIMMER_ATEM_END}"
        " suffix"
    )

    result = parse_model_format_tool_calls(
        output, _specs(_tool("allowed")), id_factory=_id_factory()
    )

    assert result == []


def test_parser_extracts_valid_calls_from_surrounding_text():
    output = (
        "prefix "
        f"{QWEN35_TOOL_CALL_START}<function=lookup></function>{QWEN35_TOOL_CALL_END}"
        f"{GEMMA4_TOOL_CALL_START}call:unknown{{}}{GEMMA4_TOOL_CALL_END}"
        " suffix"
    )

    result = parse_model_format_tool_calls(
        output, _specs(_tool("lookup")), id_factory=_id_factory()
    )

    assert [call["function"]["name"] for call in result] == ["lookup"]
