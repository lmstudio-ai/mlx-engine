from __future__ import annotations

import math
import re
from typing import Any

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
    build_openai_tool_call,
)
from mlx_engine.tool_protocols import GEMMA4_TOOL_CALL_END, GEMMA4_TOOL_CALL_START

_GEMMA4_CALL_PREFIX_RE = re.compile(r"^\s*call:\s*", re.DOTALL)
_GEMMA4_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_.$/-]+")
_GEMMA4_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_GEMMA4_STRING_DELIMITER = '<|"|>'
_PARSE_FAILURE = object()


class Gemma4ToolCallParser:
    start_marker = GEMMA4_TOOL_CALL_START

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]:
        allowed_tool_names = {tool.name for tool in tool_specs}
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
