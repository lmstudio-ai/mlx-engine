import re
from collections.abc import Iterable
from typing import Any

import mlx.core as mx

from mlx_engine.tool_protocols import (
    GEMMA4_STRING_DELIMITER,
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
_GEMMA4_CALL_PREFIX = "call:"
_GEMMA4_WHITESPACE = (" ", "\n", "\t", "\r")
# Keep normal generation cheap: for small structural states, boost allowed
# tokens instead of applying a full-vocabulary mask every decode step.
_FORCED_TOOL_LOGIT = 1e9


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
    """Guard reasoning boundaries and Gemma4 tool-call structure."""

    _STATE_NORMAL = 0
    _STATE_HEADER_ROOT = 1

    def __init__(
        self,
        *,
        reasoning_open: bool,
        reasoning_start_token_ids: tuple[int, ...],
        reasoning_end_token_ids: tuple[int, ...],
        tool_call_start_token_id: int,
        tool_call_end_token_id: int,
        call_prefix_token_ids: tuple[int, ...],
        tool_name_token_ids: tuple[tuple[int, ...], ...],
        open_brace_token_id: int,
        close_brace_token_id: int,
        string_delimiter_token_id: int,
        eos_token_ids: tuple[int, ...],
        whitespace_token_ids: tuple[int, ...],
        vocab_size: int,
    ):
        self._reasoning_open = reasoning_open
        self._reasoning_open_mx = mx.array(reasoning_open)
        self._reasoning_start_token_ids = reasoning_start_token_ids
        self._reasoning_start_first_token_id = reasoning_start_token_ids[0]
        self._reasoning_start_second_token_id = reasoning_start_token_ids[1]
        self._reasoning_end_token_ids = reasoning_end_token_ids
        self._reasoning_end_token_id = reasoning_end_token_ids[0]
        self._tool_call_start_token_id = tool_call_start_token_id
        self._tool_call_end_token_id = tool_call_end_token_id
        self._open_brace_token_id = open_brace_token_id
        self._close_brace_token_id = close_brace_token_id
        self._string_delimiter_token_id = string_delimiter_token_id
        self._tail_tokens: list[int] = []
        self._previous_token_mx: mx.array | None = None

        (
            header_edges,
            self._header_allowed_token_ids_by_state,
            self._args_state,
            self._need_tool_end_state,
            self._post_tool_state,
        ) = _build_gemma4_header_states(
            call_prefix_token_ids=call_prefix_token_ids,
            tool_name_token_ids=tool_name_token_ids,
            open_brace_token_id=open_brace_token_id,
        )
        self._header_transitions = _header_transitions(header_edges)
        self._blocked_args_token_ids = _in_vocab(
            (tool_call_start_token_id, tool_call_end_token_id, *eos_token_ids),
            vocab_size,
        )
        self._blocked_string_token_ids = _in_vocab(
            (tool_call_start_token_id, tool_call_end_token_id, *eos_token_ids),
            vocab_size,
        )
        self._post_tool_token_ids = _in_vocab(
            (*eos_token_ids, *whitespace_token_ids, tool_call_start_token_id),
            vocab_size,
        )
        self._context_token_count = 0
        self._reset_tool_state()

    def __call__(self, tokens: Any, logits: Any) -> Any:
        # VLM calls this once on prompt prefill. Prompt reasoning state already
        # came from decoded prompt text, so keep only enough tail for markers
        # that may finish immediately after prefill. Historical prompt tool
        # calls are ignored; the structural grammar starts on generated
        # <|tool_call> only.
        token_ids = _token_ids_to_list(tokens)
        self._tail_tokens = _tail(token_ids, 1)
        self._previous_token_mx = (
            mx.array(self._tail_tokens[-1]) if len(self._tail_tokens) > 0 else None
        )
        self._reasoning_open_mx = mx.array(self._reasoning_open)
        self._context_token_count = len(token_ids)
        self._reset_tool_state()
        return self._mask_if_needed(logits)

    def process_last_token_with_context(
        self,
        token_context: list[int],
        last_token: mx.array,
        logits: mx.array,
    ) -> mx.array:
        self._process_tool_context_tokens(token_context)
        return self._process_last_token_mx(last_token.reshape(-1)[0], logits)

    def process_last_token(self, last_token: Any, logits: Any) -> Any:
        if isinstance(last_token, mx.array):
            return self._process_last_token_mx(last_token.reshape(-1)[0], logits)

        for token_id in _token_ids_to_list(last_token):
            token_window = self._tail_tokens + [token_id]
            if _ends_with(token_window, self._reasoning_start_token_ids):
                self._reasoning_open = True
            if _ends_with(token_window, self._reasoning_end_token_ids):
                self._reasoning_open = False
            self._tail_tokens = _tail(token_window, 1)
        return self._mask_if_needed(logits)

    def _reset_tool_state(self) -> None:
        self._tool_state = self._STATE_NORMAL
        self._brace_depth = 0
        self._in_string = False

    def _process_tool_context_tokens(self, token_context: list[int]) -> None:
        for token_id in token_context[self._context_token_count :]:
            self._process_tool_context_token(int(token_id))
        self._context_token_count = len(token_context)

    def _process_tool_context_token(self, token_id: int) -> None:
        if self._tool_state == self._STATE_NORMAL:
            if token_id == self._tool_call_start_token_id:
                self._tool_state = self._STATE_HEADER_ROOT
            return

        if self._tool_state in self._header_transitions:
            next_state = self._header_transitions[self._tool_state].get(token_id)
            if next_state is not None:
                self._tool_state = next_state
                if next_state == self._args_state:
                    self._brace_depth = 1
                    self._in_string = False
            return

        if self._tool_state == self._args_state:
            if token_id == self._string_delimiter_token_id:
                self._in_string = not self._in_string
                return
            if self._in_string:
                return
            if token_id == self._open_brace_token_id:
                self._brace_depth += 1
                return
            if token_id == self._close_brace_token_id:
                self._brace_depth -= 1
                if self._brace_depth <= 0:
                    self._tool_state = self._need_tool_end_state
                return

        if self._tool_state == self._need_tool_end_state:
            if token_id == self._tool_call_end_token_id:
                self._tool_state = self._post_tool_state
            return

        if self._tool_state == self._post_tool_state:
            if token_id == self._tool_call_start_token_id:
                self._tool_state = self._STATE_HEADER_ROOT
                self._brace_depth = 0
                self._in_string = False

    def _process_last_token_mx(self, token_id: mx.array, logits: mx.array) -> mx.array:
        # Keep decode token handling in MLX: last_token.tolist() calls eval()/wait
        # and can create a per-token graph break.
        if self._previous_token_mx is None:
            reasoning_start = mx.array(False)
        else:
            reasoning_start = (
                self._previous_token_mx == self._reasoning_start_first_token_id
            ) & (token_id == self._reasoning_start_second_token_id)
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
        return self._mask_if_needed_mx(logits, token_id)

    def _mask_if_needed_mx(self, logits: mx.array, token_id: mx.array) -> mx.array:
        logits = self._mask_tool_structure_mx(logits, token_id)
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

    def _mask_tool_structure_mx(
        self,
        logits: mx.array,
        token_id: mx.array,
    ) -> mx.array:
        # KISS structural scope: after <|tool_call>, enforce only
        #   call:KNOWN_TOOL{ balanced top-level args }<tool_call|>
        # Argument contents remain intentionally loose; strings use Gemma's
        # <|"|> delimiter so braces inside strings do not affect balance.
        if self._tool_state == self._STATE_NORMAL:
            return _force_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._header_allowed_token_ids_by_state[self._STATE_HEADER_ROOT],
            )

        if self._tool_state in self._header_transitions:
            for edge_token_id, next_state in self._header_transitions[
                self._tool_state
            ].items():
                if next_state == self._args_state:
                    logits = self._mask_args_after_current_token(
                        logits,
                        token_id,
                        token_id == edge_token_id,
                    )
                else:
                    logits = _force_token_ids_mx(
                        logits,
                        token_id == edge_token_id,
                        self._header_allowed_token_ids_by_state[next_state],
                    )
            return logits

        if self._tool_state == self._args_state:
            return self._mask_args_after_current_token(logits, token_id, mx.array(True))

        if self._tool_state == self._need_tool_end_state:
            return _force_token_ids_mx(
                logits,
                token_id == self._tool_call_end_token_id,
                self._post_tool_token_ids,
            )

        if self._tool_state == self._post_tool_state:
            logits = _force_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._header_allowed_token_ids_by_state[self._STATE_HEADER_ROOT],
            )
            return _force_token_ids_mx(
                logits,
                token_id != self._tool_call_start_token_id,
                self._post_tool_token_ids,
            )

        return logits

    def _mask_args_after_current_token(
        self,
        logits: mx.array,
        token_id: mx.array,
        condition: mx.array,
    ) -> mx.array:
        blocked_token_ids = (
            self._blocked_string_token_ids
            if self._in_string
            else self._blocked_args_token_ids
        )
        logits = _mask_token_ids_mx(logits, condition, blocked_token_ids)
        if self._in_string or self._brace_depth != 1:
            return logits
        return _force_token_ids_mx(
            logits,
            condition & (token_id == self._close_brace_token_id),
            (self._tool_call_end_token_id,),
        )

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
    tool_call_end_token_ids = tokenizer.tool_call_end_tokens
    reasoning_start_token_ids = tokenizer.think_start_tokens
    reasoning_end_token_ids = tokenizer.think_end_tokens
    if (
        tool_call_start_token_ids is None
        or len(tool_call_start_token_ids) != 1
        or tool_call_end_token_ids is None
        or len(tool_call_end_token_ids) != 1
        or reasoning_start_token_ids is None
        or len(reasoning_start_token_ids) != 2
        or reasoning_end_token_ids is None
        or len(reasoning_end_token_ids) != 1
    ):
        return None

    call_prefix_token_ids = _encode_token_ids(tokenizer, _GEMMA4_CALL_PREFIX)
    tool_name_token_ids = tuple(
        _encode_token_ids(tokenizer, tool_name) for tool_name in context.tool_names
    )
    open_brace_token_id = _single_token_id(tokenizer, "{")
    close_brace_token_id = _single_token_id(tokenizer, "}")
    string_delimiter_token_id = _single_token_id(tokenizer, GEMMA4_STRING_DELIMITER)
    if (
        len(call_prefix_token_ids) == 0
        or any(len(token_ids) == 0 for token_ids in tool_name_token_ids)
        or open_brace_token_id is None
        or close_brace_token_id is None
        or string_delimiter_token_id is None
    ):
        return None

    vocab_size = int(tokenizer.vocab_size)
    return Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=context.reasoning_open,
        reasoning_start_token_ids=reasoning_start_token_ids,
        reasoning_end_token_ids=reasoning_end_token_ids,
        tool_call_start_token_id=tool_call_start_token_ids[0],
        tool_call_end_token_id=tool_call_end_token_ids[0],
        call_prefix_token_ids=call_prefix_token_ids,
        tool_name_token_ids=tool_name_token_ids,
        open_brace_token_id=open_brace_token_id,
        close_brace_token_id=close_brace_token_id,
        string_delimiter_token_id=string_delimiter_token_id,
        eos_token_ids=tuple(int(token_id) for token_id in tokenizer.eos_token_ids),
        whitespace_token_ids=tuple(
            token_id
            for whitespace in _GEMMA4_WHITESPACE
            if (token_id := _single_token_id(tokenizer, whitespace)) is not None
        ),
        vocab_size=vocab_size,
    )


def _build_gemma4_header_states(
    *,
    call_prefix_token_ids: tuple[int, ...],
    tool_name_token_ids: tuple[tuple[int, ...], ...],
    open_brace_token_id: int,
) -> tuple[
    tuple[tuple[int, int, int], ...],
    dict[int, tuple[int, ...]],
    int,
    int,
    int,
]:
    transitions: dict[int, dict[int, int | None]] = {}
    next_state = Gemma4ReasoningGuardLogitsProcessor._STATE_HEADER_ROOT + 1

    for name_token_ids in tool_name_token_ids:
        state = Gemma4ReasoningGuardLogitsProcessor._STATE_HEADER_ROOT
        for token_id in call_prefix_token_ids + name_token_ids:
            state_transitions = transitions.setdefault(state, {})
            if token_id not in state_transitions:
                state_transitions[token_id] = next_state
                next_state += 1
            state = int(state_transitions[token_id])
        transitions.setdefault(state, {})[open_brace_token_id] = None

    args_state = next_state
    need_tool_end_state = args_state + 1
    post_tool_state = args_state + 2

    header_edges = []
    header_allowed_token_ids_by_state = {}
    for from_state, state_transitions in transitions.items():
        header_allowed_token_ids_by_state[from_state] = tuple(state_transitions.keys())
        for token_id, to_state in state_transitions.items():
            header_edges.append(
                (from_state, token_id, args_state if to_state is None else to_state)
            )

    return (
        tuple(header_edges),
        header_allowed_token_ids_by_state,
        args_state,
        need_tool_end_state,
        post_tool_state,
    )


def _header_transitions(
    header_edges: tuple[tuple[int, int, int], ...],
) -> dict[int, dict[int, int]]:
    transitions: dict[int, dict[int, int]] = {}
    for from_state, token_id, to_state in header_edges:
        transitions.setdefault(from_state, {})[token_id] = to_state
    return transitions


def _force_token_ids_mx(
    logits: mx.array,
    condition: mx.array,
    token_ids: tuple[int, ...] | list[int],
) -> mx.array:
    if len(token_ids) == 0:
        return logits
    logits[:, list(token_ids)] = mx.where(
        condition,
        _FORCED_TOOL_LOGIT,
        logits[:, list(token_ids)],
    )
    return logits


def _mask_token_ids_mx(
    logits: mx.array,
    condition: mx.array,
    token_ids: tuple[int, ...] | list[int],
) -> mx.array:
    if len(token_ids) == 0:
        return logits
    logits[:, list(token_ids)] = mx.where(
        condition,
        -float("inf"),
        logits[:, list(token_ids)],
    )
    return logits


def _encode_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(token_id)
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    )


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    token_ids = _encode_token_ids(tokenizer, text)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def _in_vocab(token_ids: Iterable[int], vocab_size: int) -> list[int]:
    return [int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab_size]


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
