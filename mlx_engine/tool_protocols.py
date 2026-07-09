from dataclasses import dataclass

GEMMA4_REASONING_START = "<|channel>thought"
GEMMA4_CHANNEL_END = "<channel|>"
GEMMA4_TOOL_DECLARATION_START = "<|tool>"
GEMMA4_TOOL_DECLARATION_END = "<tool|>"
GEMMA4_TOOL_CALL_START = "<|tool_call>"
GEMMA4_TOOL_CALL_END = "<tool_call|>"

QWEN35_TOOLS_START = "<tools>"
QWEN35_TOOLS_END = "</tools>"
QWEN35_TOOL_CALL_START = "<tool_call>"
QWEN35_TOOL_CALL_END = "</tool_call>"
QWEN35_FUNCTION_START = "<function="
QWEN35_REASONING_START = "<think>"
QWEN35_REASONING_END = "</think>"


@dataclass(frozen=True)
class Gemma4ToolContext:
    tool_names: tuple[str, ...]
    reasoning_open: bool


@dataclass(frozen=True)
class Qwen35ToolContext:
    tool_names: tuple[str, ...]
    reasoning_open: bool


def gemma4_reasoning_is_open(text: str) -> bool:
    return text.rfind(GEMMA4_REASONING_START) > text.rfind(GEMMA4_CHANNEL_END)


def qwen35_reasoning_is_open(text: str) -> bool:
    return text.rfind(QWEN35_REASONING_START) > text.rfind(QWEN35_REASONING_END)
