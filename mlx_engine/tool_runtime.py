import re
from collections.abc import Iterable
from typing import Any

import mlx.core as mx

from mlx_engine.tool_protocols import (
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

    def __init__(
        self,
        *,
        reasoning_open: bool,
        reasoning_start_token_ids: tuple[int, ...],
        reasoning_end_token_ids: tuple[int, ...],
        tool_call_start_token_id: int,
    ):
        self._reasoning_open = reasoning_open
        self._reasoning_open_mx = mx.array(reasoning_open)
        self._reasoning_start_token_ids = reasoning_start_token_ids
        self._reasoning_start_prefix_token_id = (
            reasoning_start_token_ids[0]
            if len(reasoning_start_token_ids) == 2
            else None
        )
        self._reasoning_start_token_id = reasoning_start_token_ids[-1]
        self._reasoning_end_token_ids = reasoning_end_token_ids
        self._reasoning_end_token_id = reasoning_end_token_ids[0]
        self._tool_call_start_token_id = tool_call_start_token_id
        self._tail_token_count = max(len(reasoning_start_token_ids), 1) - 1
        self._tail_tokens: list[int] = []
        self._previous_token_mx: mx.array | None = None

    def __call__(self, tokens: Any, logits: Any) -> Any:
        # VLM calls this once on prompt prefill. Prompt reasoning state already
        # came from decoded prompt text, so keep only enough tail for markers
        # that may finish immediately after prefill.
        self._tail_tokens = _tail(_token_ids_to_list(tokens), self._tail_token_count)
        self._previous_token_mx = (
            mx.array(self._tail_tokens[-1]) if len(self._tail_tokens) > 0 else None
        )
        self._reasoning_open_mx = mx.array(self._reasoning_open)
        return self._mask_if_needed(logits)

    def process_last_token(self, last_token: Any, logits: Any) -> Any:
        if isinstance(last_token, mx.array):
            return self._process_last_token_mx(last_token.reshape(-1)[0], logits)

        for token_id in _token_ids_to_list(last_token):
            token_window = self._tail_tokens + [token_id]
            if _ends_with(token_window, self._reasoning_start_token_ids):
                self._reasoning_open = True
            if _ends_with(token_window, self._reasoning_end_token_ids):
                self._reasoning_open = False
            self._tail_tokens = _tail(token_window, self._tail_token_count)
        return self._mask_if_needed(logits)

    def _process_last_token_mx(self, token_id: mx.array, logits: mx.array) -> mx.array:
        reasoning_start = token_id == self._reasoning_start_token_id
        if self._reasoning_start_prefix_token_id is not None:
            if self._previous_token_mx is None:
                reasoning_start = mx.array(False)
            else:
                reasoning_start = reasoning_start & (
                    self._previous_token_mx == self._reasoning_start_prefix_token_id
                )
        reasoning_end = token_id == self._reasoning_end_token_id
        self._reasoning_open_mx = mx.where(
            reasoning_start,
            mx.array(True),
            self._reasoning_open_mx,
        )
        self._reasoning_open_mx = mx.where(
            reasoning_end,
            mx.array(False),
            self._reasoning_open_mx,
        )
        self._previous_token_mx = token_id
        return self._mask_if_needed_mx(logits)

    def _mask_if_needed_mx(self, logits: mx.array) -> mx.array:
        # vLLM treats <|tool_call> as an implicit Gemma4 reasoning end, but
        # Electron currently suppresses tool parsing for reasoning fragments.
        # This stricter mask forces the model to sample the real <channel|>
        # close first, avoiding synthetic markers or an Electron parser change.
        logits[:, self._tool_call_start_token_id] = mx.where(
            self._reasoning_open_mx,
            -float("inf"),
            logits[:, self._tool_call_start_token_id],
        )
        return logits

    def _mask_if_needed(self, logits: Any) -> Any:
        if self._reasoning_open:
            # vLLM treats <|tool_call> as an implicit Gemma4 reasoning end, but
            # Electron currently suppresses tool parsing for reasoning fragments.
            # This stricter mask forces the model to sample the real <channel|>
            # close first, avoiding synthetic markers or an Electron parser change.
            logits[:, self._tool_call_start_token_id] = -float("inf")
        return logits


def create_gemma4_reasoning_guard_logits_processor(
    *,
    tokenizer: Any,
    context: Gemma4ToolContext,
) -> Gemma4ReasoningGuardLogitsProcessor | None:
    if len(context.tool_names) == 0:
        return None

    tool_call_start_token_ids = tokenizer.tool_call_start_tokens
    reasoning_start_token_ids = tokenizer.think_start_tokens
    reasoning_end_token_ids = tokenizer.think_end_tokens
    if (
        tool_call_start_token_ids is None
        or len(tool_call_start_token_ids) != 1
        or reasoning_start_token_ids is None
        or not 1 <= len(reasoning_start_token_ids) <= 2
        or reasoning_end_token_ids is None
        or len(reasoning_end_token_ids) != 1
    ):
        return None

    return Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=context.reasoning_open,
        reasoning_start_token_ids=reasoning_start_token_ids,
        reasoning_end_token_ids=reasoning_end_token_ids,
        tool_call_start_token_id=tool_call_start_token_ids[0],
    )


def _token_ids_to_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, int):
        return [tokens]
    return [int(token_id) for token_id in tokens]


def _tail(token_ids: list[int], count: int) -> list[int]:
    if count == 0:
        return []
    return token_ids[-count:]


def _ends_with(token_ids: list[int], suffix: tuple[int, ...]) -> bool:
    return token_ids[-len(suffix) :] == list(suffix)
