from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from typing import Any, Callable

from mlx_engine.openai_tool_calling.model_parsers.base import ModelToolCallParser
from mlx_engine.openai_tool_calling.model_parsers.tagged_parameters import (
    iter_delimited_blocks,
)
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
    build_openai_tool_call,
)


@dataclass(frozen=True)
class UpstreamToolParserAdapter:
    module_names: tuple[str, ...]
    start_marker: str
    end_marker: str

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]:
        parse_tool_call = self._load_parse_tool_call()
        if parse_tool_call is None:
            return []

        allowed_tool_names = {tool.name for tool in tool_specs}
        template_tools = [tool.to_openai_tool() for tool in tool_specs]
        calls: list[JsonObject] = []
        for block_body in iter_delimited_blocks(
            model_output,
            self.start_marker,
            self.end_marker,
        ):
            for parsed_call in _as_list(_try_parse(parse_tool_call, block_body, template_tools)):
                normalized = _normalize_upstream_tool_call(parsed_call)
                if normalized is None:
                    continue
                tool_name, arguments = normalized
                if tool_name not in allowed_tool_names:
                    continue
                try:
                    calls.append(
                        build_openai_tool_call(
                            tool_name,
                            arguments,
                            len(calls),
                            id_factory=id_factory,
                        )
                    )
                except ValueError:
                    continue
        return calls

    def _load_parse_tool_call(self) -> Callable[[str, Any], Any] | None:
        for module_name in self.module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            parse_tool_call = getattr(module, "parse_tool_call", None)
            if callable(parse_tool_call):
                return parse_tool_call
        return None


@dataclass(frozen=True)
class FallbackToolCallParser:
    primary: ModelToolCallParser
    fallback: ModelToolCallParser

    @property
    def start_marker(self) -> str:
        return self.primary.start_marker

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]:
        calls = self.primary.parse(
            model_output,
            tool_specs,
            id_factory=id_factory,
        )
        if calls:
            return calls
        return self.fallback.parse(
            model_output,
            tool_specs,
            id_factory=id_factory,
        )


def _try_parse(
    parse_tool_call: Callable[[str, Any], Any],
    block_body: str,
    tools: list[JsonObject],
) -> Any | None:
    try:
        return parse_tool_call(block_body, tools)
    except Exception:
        return None


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize_upstream_tool_call(value: Any) -> tuple[str, JsonObject] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or name == "":
        return None
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return name, arguments
