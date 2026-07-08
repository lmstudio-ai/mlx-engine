from dataclasses import dataclass

GEMMA4_REASONING_START = "<|channel>thought"
GEMMA4_CHANNEL_END = "<channel|>"
GEMMA4_TOOL_DECLARATION_START = "<|tool>"
GEMMA4_TOOL_DECLARATION_END = "<tool|>"
GEMMA4_TOOL_CALL_START = "<|tool_call>"
GEMMA4_TOOL_CALL_END = "<tool_call|>"
GEMMA4_STRING_DELIMITER = '<|"|>'


@dataclass(frozen=True)
class Gemma4ToolContext:
    tool_names: tuple[str, ...]
    reasoning_open: bool


def gemma4_reasoning_is_open(text: str) -> bool:
    return text.rfind(GEMMA4_REASONING_START) > text.rfind(GEMMA4_CHANNEL_END)
