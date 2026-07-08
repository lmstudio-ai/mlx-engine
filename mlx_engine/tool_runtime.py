import re
from collections.abc import Iterable
from typing import Any

from mlx_engine.tool_protocols import (
    GEMMA4_TOOL_CALL_START,
    GEMMA4_TOOL_DECLARATION_END,
    GEMMA4_TOOL_DECLARATION_START,
    Gemma4ToolContext,
    gemma4_reasoning_is_open,
)

# Match Gemma4 declaration blocks like:
#   <|tool>declaration:get_weather{ ... }<tool|>
# and capture only the declared tool name (`get_weather`).
_GEMMA4_TOOL_NAME_RE = re.compile(
    rf"{re.escape(GEMMA4_TOOL_DECLARATION_START)}.*?"
    r"declaration:\s*([A-Za-z_][A-Za-z0-9_.$/-]*)\s*{.*?"
    rf"{re.escape(GEMMA4_TOOL_DECLARATION_END)}",
    re.DOTALL,
)


def create_gemma4_tool_context_from_prompt(
    *,
    tokenizer: Any,
    prompt_tokens: Iterable[int],
    model_type: str | None,
) -> Gemma4ToolContext | None:
    """Return Gemma4 tool context only when the rendered prompt declares tools."""
    if not str(model_type or "").startswith("gemma4"):
        return None

    prompt_text = tokenizer.decode(list(prompt_tokens))
    tool_names = tuple(dict.fromkeys(_GEMMA4_TOOL_NAME_RE.findall(prompt_text)))
    if len(tool_names) == 0:
        return None

    return Gemma4ToolContext(
        tool_names=tool_names,
        reasoning_open=gemma4_reasoning_is_open(prompt_text),
    )


class Gemma4ReasoningGuardLogitsProcessor:
    """Block Gemma4 tool-call starts while visible reasoning is still open."""

    def __init__(self, *, tokenizer: Any, tool_call_start_token_id: int):
        self._tokenizer = tokenizer
        self._tool_call_start_token_id = tool_call_start_token_id

    def __call__(self, tokens: Any, logits: Any) -> Any:
        token_ids = _token_ids_to_list(tokens)
        if not gemma4_reasoning_is_open(self._tokenizer.decode(token_ids)):
            return logits

        if 0 <= self._tool_call_start_token_id < logits.shape[-1]:
            logits[:, self._tool_call_start_token_id] = -float("inf")
        return logits


def create_gemma4_reasoning_guard_logits_processor(
    *,
    tokenizer: Any,
    context: Gemma4ToolContext,
) -> Gemma4ReasoningGuardLogitsProcessor | None:
    if len(context.tool_names) == 0:
        return None

    tool_call_start_token_id = _gemma4_tool_call_start_token_id(tokenizer)
    if tool_call_start_token_id is None:
        return None

    return Gemma4ReasoningGuardLogitsProcessor(
        tokenizer=tokenizer,
        tool_call_start_token_id=tool_call_start_token_id,
    )


def _token_ids_to_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, int):
        return [tokens]
    return [int(token_id) for token_id in tokens]


def _gemma4_tool_call_start_token_id(tokenizer: Any) -> int | None:
    token_ids = getattr(tokenizer, "tool_call_start_tokens", None)
    if token_ids is None and hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(GEMMA4_TOOL_CALL_START, add_special_tokens=False)

    if isinstance(token_ids, int):
        token_id = token_ids
    elif isinstance(token_ids, (list, tuple)) and len(token_ids) == 1:
        token_id = int(token_ids[0])
    else:
        return None

    try:
        if tokenizer.decode([token_id]) != GEMMA4_TOOL_CALL_START:
            return None
    except Exception:
        return None
    return token_id
