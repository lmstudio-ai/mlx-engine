from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Pattern

from mlx_engine.openai_tool_calling.model_parsers.base import parse_json_argument_value
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
    build_openai_tool_call,
)


@dataclass(frozen=True)
class TaggedParameterToolCallParser:
    start_marker: str
    end_marker: str
    invocation_pattern: Pattern[str]
    parameter_pattern: Pattern[str]

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]:
        allowed_tool_names = {tool.name for tool in tool_specs}
        calls: list[JsonObject] = []
        for block_body in iter_delimited_blocks(
            model_output,
            self.start_marker,
            self.end_marker,
        ):
            for invoke_match in self.invocation_pattern.finditer(block_body):
                tool_name = invoke_match.group(1).strip()
                if tool_name not in allowed_tool_names:
                    continue
                arguments: JsonObject = {}
                for parameter_match in self.parameter_pattern.finditer(
                    invoke_match.group(2)
                ):
                    parameter_name = parameter_match.group(1).strip()
                    if parameter_name == "":
                        continue
                    arguments[parameter_name] = parse_json_argument_value(
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


def iter_delimited_blocks(
    text: str,
    start_marker: str,
    end_marker: str,
):
    block_pattern = re.compile(
        rf"{re.escape(start_marker)}(.*?){re.escape(end_marker)}",
        re.DOTALL,
    )
    for block_match in block_pattern.finditer(text):
        yield block_match.group(1)
