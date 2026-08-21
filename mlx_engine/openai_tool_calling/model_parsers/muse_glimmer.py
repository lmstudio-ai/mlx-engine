from __future__ import annotations

import re

from mlx_engine.openai_tool_calling.model_parsers.tagged_parameters import (
    TaggedParameterToolCallParser,
)
from mlx_engine.tool_protocols import MUSE_GLIMMER_ATEM_END, MUSE_GLIMMER_ATEM_START


class MuseGlimmerToolCallParser(TaggedParameterToolCallParser):
    def __init__(self):
        super().__init__(
            start_marker=MUSE_GLIMMER_ATEM_START,
            end_marker=MUSE_GLIMMER_ATEM_END,
            invocation_pattern=re.compile(
                r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>',
                re.DOTALL,
            ),
            parameter_pattern=re.compile(
                r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>',
                re.DOTALL,
            ),
        )
