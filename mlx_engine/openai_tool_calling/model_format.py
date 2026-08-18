from __future__ import annotations

import json
import re
from typing import Any

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ParsedToolCalls,
    ToolCallIdFactory,
    ToolCallingValidationError,
    build_openai_tool_call,
)
from mlx_engine.tool_protocols import (
    GEMMA4_TOOL_CALL_END,
    GEMMA4_TOOL_CALL_START,
    MUSE_GLIMMER_ATEM_END,
    MUSE_GLIMMER_ATEM_START,
    QWEN35_TOOL_CALL_END,
    QWEN35_TOOL_CALL_START,
)

_QWEN35_BLOCK_RE = re.compile(
    rf"{re.escape(QWEN35_TOOL_CALL_START)}(.*?){re.escape(QWEN35_TOOL_CALL_END)}",
    re.DOTALL,
)
_QWEN35_FUNCTION_RE = re.compile(
    r"<function=([^>]+)>(.*?)</function>",
    re.DOTALL,
)
_QWEN35_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_GEMMA4_BLOCK_RE = re.compile(
    rf"{re.escape(GEMMA4_TOOL_CALL_START)}(.*?){re.escape(GEMMA4_TOOL_CALL_END)}",
    re.DOTALL,
)
_GEMMA4_CALL_RE = re.compile(
    r"^\s*call:\s*([A-Za-z_][A-Za-z0-9_.$/:-]*)\s*(.*)$", re.DOTALL
)
_MUSE_GLIMMER_BLOCK_RE = re.compile(
    rf"{re.escape(MUSE_GLIMMER_ATEM_START)}(.*?){re.escape(MUSE_GLIMMER_ATEM_END)}",
    re.DOTALL,
)
_MUSE_GLIMMER_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>', re.DOTALL
)
_MUSE_GLIMMER_PARAMETER_RE = re.compile(
    r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL
)
_GEMMA4_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_.$/-]+")
_GEMMA4_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_GEMMA4_STRING_DELIMITER = '<|"|>'
MODEL_FORMAT_TOOL_CALL_START_MARKERS = (
    QWEN35_TOOL_CALL_START,
    GEMMA4_TOOL_CALL_START,
    MUSE_GLIMMER_ATEM_START,
)


def parse_model_format_tool_calls(
    model_output: str,
    tool_specs: list[FunctionToolSpec],
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> ParsedToolCalls:
    """Parse supported model-format MLX tool-call text into OpenAI tool calls."""
    allowed_tool_names = {tool.name for tool in tool_specs}

    qwen35_result = parse_qwen35_tool_calls(
        model_output,
        allowed_tool_names,
        starting_index=0,
        id_factory=id_factory,
    )
    gemma4_result = parse_gemma4_tool_calls(
        qwen35_result.remaining_text,
        allowed_tool_names,
        starting_index=len(qwen35_result.calls),
        id_factory=id_factory,
    )
    muse_glimmer_result = parse_muse_glimmer_tool_calls(
        gemma4_result.remaining_text,
        allowed_tool_names,
        starting_index=len(qwen35_result.calls) + len(gemma4_result.calls),
        id_factory=id_factory,
    )
    _reject_mixed_model_formats(
        qwen35_calls=len(qwen35_result.calls),
        gemma4_calls=len(gemma4_result.calls),
        muse_glimmer_calls=len(muse_glimmer_result.calls),
    )
    return ParsedToolCalls(
        calls=[
            *qwen35_result.calls,
            *gemma4_result.calls,
            *muse_glimmer_result.calls,
        ],
        remaining_text=muse_glimmer_result.remaining_text,
    )


def _reject_mixed_model_formats(
    *,
    qwen35_calls: int,
    gemma4_calls: int,
    muse_glimmer_calls: int,
) -> None:
    used_formats = sum(
        call_count > 0
        for call_count in (qwen35_calls, gemma4_calls, muse_glimmer_calls)
    )
    if used_formats > 1:
        raise ToolCallingValidationError(
            "Mixed model-format tool calls are not supported in one response."
        )


def parse_qwen35_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    starting_index: int = 0,
    id_factory: ToolCallIdFactory | None = None,
) -> ParsedToolCalls:
    calls: list[JsonObject] = []
    converted_block_spans: list[tuple[int, int]] = []
    for block_match in _QWEN35_BLOCK_RE.finditer(model_output):
        block_call_count = len(calls)
        block_body = block_match.group(1)
        for function_match in _QWEN35_FUNCTION_RE.finditer(block_body):
            tool_name = function_match.group(1).strip()
            if tool_name not in allowed_tool_names:
                continue
            arguments: JsonObject = {}
            for parameter_match in _QWEN35_PARAMETER_RE.finditer(
                function_match.group(2)
            ):
                parameter_name = parameter_match.group(1).strip()
                if parameter_name == "":
                    continue
                arguments[parameter_name] = parse_qwen35_tool_argument_value(
                    parameter_match.group(2)
                )
            calls.append(
                build_openai_tool_call(
                    tool_name,
                    arguments,
                    starting_index + len(calls),
                    id_factory=id_factory,
                )
            )
        if len(calls) > block_call_count:
            converted_block_spans.append(block_match.span())

    return ParsedToolCalls(
        calls=calls,
        remaining_text=_remove_spans(model_output, converted_block_spans),
    )


def parse_qwen35_tool_argument_value(value: str) -> Any:
    stripped_value = value.strip()
    if stripped_value == "":
        return ""
    try:
        return json.loads(stripped_value)
    except json.JSONDecodeError:
        return stripped_value


def parse_gemma4_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    starting_index: int = 0,
    id_factory: ToolCallIdFactory | None = None,
) -> ParsedToolCalls:
    calls: list[JsonObject] = []
    converted_block_spans: list[tuple[int, int]] = []
    for block_match in _GEMMA4_BLOCK_RE.finditer(model_output):
        call_match = _GEMMA4_CALL_RE.match(block_match.group(1))
        if call_match is None:
            continue
        tool_name = call_match.group(1).strip()
        if tool_name not in allowed_tool_names:
            continue
        arguments = parse_gemma4_arguments_object(call_match.group(2).strip())
        if arguments is None:
            continue
        calls.append(
            build_openai_tool_call(
                tool_name,
                arguments,
                starting_index + len(calls),
                id_factory=id_factory,
            )
        )
        converted_block_spans.append(block_match.span())

    return ParsedToolCalls(
        calls=calls,
        remaining_text=_remove_spans(model_output, converted_block_spans),
    )


def parse_muse_glimmer_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    starting_index: int = 0,
    id_factory: ToolCallIdFactory | None = None,
) -> ParsedToolCalls:
    calls: list[JsonObject] = []
    converted_block_spans: list[tuple[int, int]] = []
    for block_match in _MUSE_GLIMMER_BLOCK_RE.finditer(model_output):
        block_call_count = len(calls)
        for invoke_match in _MUSE_GLIMMER_INVOKE_RE.finditer(block_match.group(1)):
            tool_name = invoke_match.group(1).strip()
            if tool_name not in allowed_tool_names:
                continue
            arguments: JsonObject = {}
            for parameter_match in _MUSE_GLIMMER_PARAMETER_RE.finditer(
                invoke_match.group(2)
            ):
                parameter_name = parameter_match.group(1).strip()
                if parameter_name == "":
                    continue
                arguments[parameter_name] = parse_qwen35_tool_argument_value(
                    parameter_match.group(2)
                )
            calls.append(
                build_openai_tool_call(
                    tool_name,
                    arguments,
                    starting_index + len(calls),
                    id_factory=id_factory,
                )
            )
        if len(calls) > block_call_count:
            converted_block_spans.append(block_match.span())

    return ParsedToolCalls(
        calls=calls,
        remaining_text=_remove_spans(model_output, converted_block_spans),
    )


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if len(spans) == 0:
        return text.strip()

    parts: list[str] = []
    position = 0
    for start, end in spans:
        parts.append(text[position:start])
        parts.append(" ")
        position = end
    parts.append(text[position:])
    return "".join(parts).strip()


def parse_gemma4_arguments_object(value: str) -> JsonObject | None:
    parser = _Gemma4ValueParser(value)
    parsed_value = parser.parse_object()
    if parsed_value is None:
        return None
    parser.skip_whitespace()
    if not parser.at_end():
        return None
    return parsed_value


class _Gemma4ValueParser:
    def __init__(self, text: str):
        self._text = text
        self._position = 0

    def at_end(self) -> bool:
        return self._position >= len(self._text)

    def skip_whitespace(self) -> None:
        while not self.at_end() and self._text[self._position] in " \t\n\r":
            self._position += 1

    def parse_object(self) -> JsonObject | None:
        self.skip_whitespace()
        if not self._consume("{"):
            return None
        result: JsonObject = {}
        self.skip_whitespace()
        if self._consume("}"):
            return result

        while True:
            key = self._parse_key()
            if key is None:
                return None
            self.skip_whitespace()
            if not self._consume(":"):
                return None
            value = self._parse_value()
            if value is _PARSE_FAILURE:
                return None
            result[key] = value
            self.skip_whitespace()
            if self._consume("}"):
                return result
            if not self._consume(","):
                return None
            self.skip_whitespace()

    def _parse_key(self) -> str | None:
        self.skip_whitespace()
        string_key = self._parse_gemma_string()
        if string_key is not None:
            return string_key

        match = _GEMMA4_BARE_KEY_RE.match(self._text, self._position)
        if match is None:
            return None
        self._position = match.end()
        return match.group(0)

    def _parse_value(self) -> Any:
        self.skip_whitespace()
        string_value = self._parse_gemma_string()
        if string_value is not None:
            return string_value

        if self._peek() == "{":
            object_value = self.parse_object()
            return _PARSE_FAILURE if object_value is None else object_value
        if self._peek() == "[":
            return self._parse_array()

        literal_value = self._parse_literal()
        if literal_value is not _PARSE_FAILURE:
            return literal_value

        number_value = self._parse_number()
        if number_value is not _PARSE_FAILURE:
            return number_value

        return _PARSE_FAILURE

    def _parse_array(self) -> Any:
        if not self._consume("["):
            return _PARSE_FAILURE
        result: list[Any] = []
        self.skip_whitespace()
        if self._consume("]"):
            return result

        while True:
            value = self._parse_value()
            if value is _PARSE_FAILURE:
                return _PARSE_FAILURE
            result.append(value)
            self.skip_whitespace()
            if self._consume("]"):
                return result
            if not self._consume(","):
                return _PARSE_FAILURE
            self.skip_whitespace()

    def _parse_gemma_string(self) -> str | None:
        if not self._consume(_GEMMA4_STRING_DELIMITER):
            return None
        end_index = self._text.find(_GEMMA4_STRING_DELIMITER, self._position)
        if end_index < 0:
            return None
        value = self._text[self._position : end_index]
        self._position = end_index + len(_GEMMA4_STRING_DELIMITER)
        return value

    def _parse_literal(self) -> Any:
        for literal_text, literal_value in (
            ("true", True),
            ("false", False),
            ("null", None),
            ("None", None),
            ("none", None),
        ):
            if self._starts_with_word(literal_text):
                self._position += len(literal_text)
                return literal_value
        return _PARSE_FAILURE

    def _parse_number(self) -> Any:
        match = _GEMMA4_NUMBER_RE.match(self._text, self._position)
        if match is None:
            return _PARSE_FAILURE
        raw_number = match.group(0)
        self._position = match.end()
        if "." in raw_number or "e" in raw_number or "E" in raw_number:
            return float(raw_number)
        return int(raw_number)

    def _peek(self) -> str | None:
        return None if self.at_end() else self._text[self._position]

    def _consume(self, expected: str) -> bool:
        if not self._text.startswith(expected, self._position):
            return False
        self._position += len(expected)
        return True

    def _starts_with_word(self, word: str) -> bool:
        if not self._text.startswith(word, self._position):
            return False
        next_position = self._position + len(word)
        if next_position >= len(self._text):
            return True
        next_char = self._text[next_position]
        return not (next_char.isalnum() or next_char in "_.$/-")


_PARSE_FAILURE = object()
