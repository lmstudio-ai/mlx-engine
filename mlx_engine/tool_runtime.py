import re
from collections.abc import Iterable
from typing import Any

from mlx_engine.tool_protocols import (
    GEMMA4_CHANNEL_END,
    GEMMA4_REASONING_START,
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

    uses_last_token_fast_path = True

    def __init__(
        self,
        *,
        reasoning_open: bool,
        reasoning_start_token_ids: tuple[int, ...],
        reasoning_end_token_ids: tuple[int, ...],
        tool_call_start_token_id: int,
    ):
        self._reasoning_open = reasoning_open
        self._reasoning_start_token_ids = reasoning_start_token_ids
        self._reasoning_end_token_ids = reasoning_end_token_ids
        self._tool_call_start_token_id = tool_call_start_token_id
        self._processed_token_count = 0
        self._tail_tokens: list[int] = []
        self._tail_token_count = max(
            len(reasoning_start_token_ids),
            len(reasoning_end_token_ids),
            1,
        ) - 1

    def __call__(self, tokens: Any, logits: Any) -> Any:
        self._sync_from_full_history(_token_ids_to_list(tokens))
        return self._mask_if_needed(logits)

    def process_last_token(self, last_token: Any, logits: Any) -> Any:
        new_token_ids = _token_ids_to_list(last_token)
        self._process_new_tokens(new_token_ids)
        self._processed_token_count += len(new_token_ids)
        return self._mask_if_needed(logits)

    def _sync_from_full_history(self, token_ids: list[int]) -> None:
        if self._processed_token_count == 0:
            # First call is the prompt prefill. We already decoded the prompt once
            # to establish this state, so do not decode or rescan it per token.
            self._processed_token_count = len(token_ids)
            self._tail_tokens = self._tail(token_ids)
            return

        if len(token_ids) < self._processed_token_count:
            self._reasoning_open = _gemma4_reasoning_is_open_from_token_ids(
                token_ids,
                self._reasoning_start_token_ids,
                self._reasoning_end_token_ids,
            )
            self._processed_token_count = len(token_ids)
            self._tail_tokens = self._tail(token_ids)
            return

        self._process_new_tokens(token_ids[self._processed_token_count :])
        self._processed_token_count = len(token_ids)

    def _process_new_tokens(self, new_token_ids: list[int]) -> None:
        if len(new_token_ids) == 0:
            return
        scan_token_ids = self._tail_tokens + new_token_ids
        self._reasoning_open = _update_gemma4_reasoning_state_from_token_ids(
            self._reasoning_open,
            scan_token_ids,
            self._reasoning_start_token_ids,
            self._reasoning_end_token_ids,
        )
        self._tail_tokens = self._tail(scan_token_ids)

    def _tail(self, token_ids: list[int]) -> list[int]:
        if self._tail_token_count == 0:
            return []
        return token_ids[-self._tail_token_count :]

    def _mask_if_needed(self, logits: Any) -> Any:
        if not self._reasoning_open:
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
    reasoning_start_token_ids = _token_sequence(
        tokenizer,
        attr_name="think_start_tokens",
        text=GEMMA4_REASONING_START,
    )
    reasoning_end_token_ids = _token_sequence(
        tokenizer,
        attr_name="think_end_tokens",
        text=GEMMA4_CHANNEL_END,
    )
    if (
        tool_call_start_token_id is None
        or reasoning_start_token_ids is None
        or reasoning_end_token_ids is None
    ):
        return None

    return Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=context.reasoning_open,
        reasoning_start_token_ids=reasoning_start_token_ids,
        reasoning_end_token_ids=reasoning_end_token_ids,
        tool_call_start_token_id=tool_call_start_token_id,
    )


def _token_ids_to_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, int):
        return [tokens]

    token_ids = []
    for token_id in tokens:
        if isinstance(token_id, (list, tuple)):
            token_ids.extend(_token_ids_to_list(token_id))
        else:
            token_ids.append(int(token_id))
    return token_ids


def _gemma4_tool_call_start_token_id(tokenizer: Any) -> int | None:
    token_ids = _token_sequence(
        tokenizer,
        attr_name="tool_call_start_tokens",
        text=GEMMA4_TOOL_CALL_START,
    )
    if token_ids is None or len(token_ids) != 1:
        return None

    token_id = token_ids[0]
    try:
        if tokenizer.decode([token_id]) != GEMMA4_TOOL_CALL_START:
            return None
    except Exception:
        return None
    return token_id


def _token_sequence(
    tokenizer: Any,
    *,
    attr_name: str,
    text: str,
) -> tuple[int, ...] | None:
    token_ids = getattr(tokenizer, attr_name, None)
    if token_ids is None and hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    if isinstance(token_ids, int):
        return (token_ids,)
    if isinstance(token_ids, (list, tuple)) and len(token_ids) > 0:
        return tuple(int(token_id) for token_id in token_ids)
    return None


def _gemma4_reasoning_is_open_from_token_ids(
    token_ids: list[int],
    reasoning_start_token_ids: tuple[int, ...],
    reasoning_end_token_ids: tuple[int, ...],
) -> bool:
    return _update_gemma4_reasoning_state_from_token_ids(
        False,
        token_ids,
        reasoning_start_token_ids,
        reasoning_end_token_ids,
    )


def _update_gemma4_reasoning_state_from_token_ids(
    reasoning_open: bool,
    token_ids: list[int],
    reasoning_start_token_ids: tuple[int, ...],
    reasoning_end_token_ids: tuple[int, ...],
) -> bool:
    for token_index in range(len(token_ids)):
        if _token_sequence_matches(token_ids, token_index, reasoning_start_token_ids):
            reasoning_open = True
        if _token_sequence_matches(token_ids, token_index, reasoning_end_token_ids):
            reasoning_open = False
    return reasoning_open


def _token_sequence_matches(
    token_ids: list[int],
    start_index: int,
    expected: tuple[int, ...],
) -> bool:
    end_index = start_index + len(expected)
    return end_index <= len(token_ids) and tuple(token_ids[start_index:end_index]) == expected
