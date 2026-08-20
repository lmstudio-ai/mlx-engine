from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
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
_GEMMA4_CALL_PREFIX_RE = re.compile(r"^\s*call:\s*", re.DOTALL)
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
ModelToolCallFormat = Literal["auto", "qwen35", "gemma4", "muse_glimmer"]
_SelectedModelToolCallFormat = Literal["qwen35", "gemma4", "muse_glimmer"]
_MODEL_FORMAT_MARKERS: tuple[tuple[_SelectedModelToolCallFormat, str], ...] = (
    ("qwen35", QWEN35_TOOL_CALL_START),
    ("gemma4", GEMMA4_TOOL_CALL_START),
    ("muse_glimmer", MUSE_GLIMMER_ATEM_START),
)


def parse_model_format_tool_calls(
    model_output: str,
    tool_specs: list[FunctionToolSpec],
    *,
    id_factory: ToolCallIdFactory | None = None,
    model_format: ModelToolCallFormat = "auto",
) -> list[JsonObject]:
    """Parse supported model-format MLX tool-call text into OpenAI tool calls."""
    allowed_tool_names = {tool.name for tool in tool_specs}
    selected_format = _select_model_tool_call_format(model_output, model_format)
    if selected_format is None:
        return []
    if selected_format == "qwen35":
        return parse_qwen35_tool_calls(
            model_output,
            allowed_tool_names,
            id_factory=id_factory,
        )
    if selected_format == "gemma4":
        return parse_gemma4_tool_calls(
            model_output,
            allowed_tool_names,
            id_factory=id_factory,
        )
    return parse_muse_glimmer_tool_calls(
        model_output,
        allowed_tool_names,
        id_factory=id_factory,
    )


def _select_model_tool_call_format(
    model_output: str,
    model_format: ModelToolCallFormat,
) -> _SelectedModelToolCallFormat | None:
    if model_format != "auto":
        return model_format
    marker_positions = [
        (position, marker_format)
        for marker_format, marker in _MODEL_FORMAT_MARKERS
        if (position := model_output.find(marker)) >= 0
    ]
    if len(marker_positions) == 0:
        return None
    return min(marker_positions)[1]


def parse_qwen35_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> list[JsonObject]:
    calls: list[JsonObject] = []
    for block_match in _QWEN35_BLOCK_RE.finditer(model_output):
        for function_match in _QWEN35_FUNCTION_RE.finditer(block_match.group(1)):
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
                    len(calls),
                    id_factory=id_factory,
                )
            )

    return calls


def parse_qwen35_tool_argument_value(value: str) -> Any:
    stripped_value = value.strip()
    if stripped_value == "":
        return ""
    try:
        return json.loads(
            stripped_value,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except ValueError:
        return stripped_value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Unsupported JSON constant: {value}")


def _parse_json_float(value: str) -> float | str:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else value


def parse_gemma4_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> list[JsonObject]:
    calls: list[JsonObject] = []
    position = 0
    while True:
        block_start = model_output.find(GEMMA4_TOOL_CALL_START, position)
        if block_start < 0:
            return calls
        body_start = block_start + len(GEMMA4_TOOL_CALL_START)
        parsed_block = _parse_gemma4_block_at(
            model_output,
            body_start,
            allowed_tool_names,
        )
        if parsed_block is None:
            position = body_start
            continue
        tool_name, arguments, block_end = parsed_block
        calls.append(
            build_openai_tool_call(
                tool_name,
                arguments,
                len(calls),
                id_factory=id_factory,
            )
        )
        position = block_end


def _parse_gemma4_block_at(
    model_output: str,
    body_start: int,
    allowed_tool_names: set[str],
) -> tuple[str, JsonObject, int] | None:
    end_search_position = body_start
    while True:
        block_end = model_output.find(GEMMA4_TOOL_CALL_END, end_search_position)
        if block_end < 0:
            return None
        # Gemma native call delimiters are protocol-reserved. We scan each
        # candidate end marker and accept the first candidate that parses,
        # including tolerant recovery for an unclosed Gemma string before the
        # marker. This intentionally favors repairing a model that meant to end
        # the tool call over preserving literal delimiter text in arguments.
        block_body = model_output[body_start:block_end]
        split_call = _split_gemma4_tool_call(block_body, allowed_tool_names)
        if split_call is not None:
            tool_name, arguments_text = split_call
            arguments = parse_gemma4_arguments_object(arguments_text.strip())
            if arguments is not None:
                return tool_name, arguments, block_end + len(GEMMA4_TOOL_CALL_END)
        end_search_position = block_end + len(GEMMA4_TOOL_CALL_END)


def _split_gemma4_tool_call(
    block_body: str,
    allowed_tool_names: set[str],
) -> tuple[str, str] | None:
    prefix_match = _GEMMA4_CALL_PREFIX_RE.match(block_body)
    if prefix_match is None:
        return None
    call_body = block_body[prefix_match.end() :]
    for tool_name in sorted(allowed_tool_names, key=len, reverse=True):
        if call_body.startswith(tool_name):
            return tool_name, call_body[len(tool_name) :]
    return None


def parse_muse_glimmer_tool_calls(
    model_output: str,
    allowed_tool_names: set[str],
    *,
    id_factory: ToolCallIdFactory | None = None,
) -> list[JsonObject]:
    calls: list[JsonObject] = []
    for block_match in _MUSE_GLIMMER_BLOCK_RE.finditer(model_output):
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
                    len(calls),
                    id_factory=id_factory,
                )
            )

    return calls


def parse_gemma4_arguments_object(value: str) -> JsonObject | None:
    for recover_unclosed_strings in (False, True):
        parser = _Gemma4ValueParser(
            value,
            recover_unclosed_strings=recover_unclosed_strings,
        )
        parsed_value = parser.parse_object()
        if parsed_value is not None:
            parser.skip_whitespace()
            if parser.at_end():
                return parsed_value
    return None


class _Gemma4ValueParser:
    def __init__(self, text: str, *, recover_unclosed_strings: bool):
        self._text = text
        self._position = 0
        self._recover_unclosed_strings = recover_unclosed_strings

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
        string_key = self._parse_gemma_string(recovery_terminators=":")
        if string_key is _PARSE_FAILURE:
            return None
        if string_key is not None:
            return string_key

        match = _GEMMA4_BARE_KEY_RE.match(self._text, self._position)
        if match is None:
            return None
        self._position = match.end()
        return match.group(0)

    def _parse_value(self) -> Any:
        self.skip_whitespace()
        string_value = self._parse_gemma_string(recovery_terminators=",}]")
        if string_value is _PARSE_FAILURE:
            return _PARSE_FAILURE
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

    def _parse_gemma_string(self, *, recovery_terminators: str) -> Any:
        if not self._consume(_GEMMA4_STRING_DELIMITER):
            return None
        value_start = self._position
        end_index = self._text.find(_GEMMA4_STRING_DELIMITER, self._position)
        recovery_end = self._find_first(recovery_terminators)
        if self._recover_unclosed_strings and (
            end_index < 0 or 0 <= recovery_end < end_index
        ):
            if recovery_end < 0:
                recovery_end = len(self._text)
            value = self._text[value_start:recovery_end]
            self._position = recovery_end
            return value
        if end_index >= 0:
            value = self._text[value_start:end_index]
            self._position = end_index + len(_GEMMA4_STRING_DELIMITER)
            return value
        return _PARSE_FAILURE

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
            parsed_float = float(raw_number)
            return parsed_float if math.isfinite(parsed_float) else raw_number
        return int(raw_number)

    def _peek(self) -> str | None:
        return None if self.at_end() else self._text[self._position]

    def _find_first(self, candidates: str) -> int:
        indexes = [
            index
            for candidate in candidates
            if (index := self._text.find(candidate, self._position)) >= 0
        ]
        return min(indexes) if indexes else -1

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
