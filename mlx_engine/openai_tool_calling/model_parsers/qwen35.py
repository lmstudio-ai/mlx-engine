from __future__ import annotations

import re

from mlx_engine.openai_tool_calling.model_parsers.tagged_parameters import (
    TaggedParameterToolCallParser,
)
from mlx_engine.tool_protocols import QWEN35_TOOL_CALL_END, QWEN35_TOOL_CALL_START


class Qwen35ToolCallParser(TaggedParameterToolCallParser):
    def __init__(self):
        super().__init__(
            start_marker=QWEN35_TOOL_CALL_START,
            end_marker=QWEN35_TOOL_CALL_END,
            invocation_pattern=re.compile(
                r"<function=([^>]+)>(.*?)</function>",
                re.DOTALL,
            ),
            parameter_pattern=re.compile(
                r"<parameter=([^>]+)>(.*?)</parameter>",
                re.DOTALL,
            ),
        )
