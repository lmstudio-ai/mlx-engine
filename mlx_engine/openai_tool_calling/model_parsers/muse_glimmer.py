from __future__ import annotations

import re

from mlx_engine.openai_tool_calling.model_parsers.base import parse_json_argument_value
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
    build_openai_tool_call,
)
from mlx_engine.tool_protocols import MUSE_GLIMMER_ATEM_END, MUSE_GLIMMER_ATEM_START

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


class MuseGlimmerToolCallParser:
    start_marker = MUSE_GLIMMER_ATEM_START

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]:
        allowed_tool_names = {tool.name for tool in tool_specs}
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
