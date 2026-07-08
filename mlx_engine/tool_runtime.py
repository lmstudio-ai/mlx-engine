import json
import re
from typing import Any

import llguidance
import llguidance.hf
import llguidance.mlx
import llguidance.numpy

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
_GEMMA4_CALL_PREFIX = "call:"
_GEMMA4_WHITESPACE = (" ", "\n", "\t", "\r")
# For tiny forced-token states, update only the allowed token ids instead of
# adding an unnecessary full-vocabulary mask.
_FORCED_TOOL_LOGIT = 1e9
_LLG_TOKENIZER_CACHE: dict[int, Any] = {}


def create_gemma4_tool_context_from_prompt(
    *,
    tokenizer: Any,
    prompt_tokens: list[int],
    model_type: str | None,
) -> Gemma4ToolContext | None:
    """Return Gemma4 tool context only when the rendered prompt declares tools."""
    if not str(model_type or "").startswith("gemma4"):
        return None

    prompt_text = tokenizer.decode(prompt_tokens)
    tool_names = tuple(dict.fromkeys(_GEMMA4_TOOL_NAME_RE.findall(prompt_text)))
    if len(tool_names) == 0:
        return None

    return Gemma4ToolContext(
        tool_names=tool_names,
        reasoning_open=gemma4_reasoning_is_open(prompt_text),
    )


class _Gemma4LLGuidanceToolGrammar:
    """llguidance-backed grammar for one Gemma4 native tool call."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        tool_names: tuple[str, ...],
        call_prefix_token_ids: tuple[int, ...],
        vocab_size: int,
    ):
        self.initial_token_ids = (call_prefix_token_ids[0],)
        self._tokenizer = tokenizer
        self._grammar = _gemma4_llguidance_grammar(tool_names)
        self._vocab_size = vocab_size
        self._llg_tokenizer = None
        self._bitmask = None

    def _ensure_ready(self) -> None:
        if self._llg_tokenizer is not None:
            return
        self._llg_tokenizer = _get_llguidance_tokenizer(self._tokenizer)
        self._bitmask = llguidance.numpy.allocate_token_bitmask(1, self._vocab_size)

    def start_matcher(self) -> Any:
        self._ensure_ready()
        return llguidance.LLMatcher(self._llg_tokenizer, self._grammar)

    def consume_token(self, matcher: Any, token_id: int) -> bool:
        matcher.consume_token(token_id)
        if error := matcher.get_error():
            raise ValueError(error)
        return matcher.is_stopped()

    def mask_logits(self, matcher: Any, logits: mx.array) -> mx.array:
        llguidance.numpy.fill_next_token_bitmask(matcher, self._bitmask, 0)
        return llguidance.mlx.apply_token_bitmask(logits, self._bitmask)


class Gemma4ReasoningGuardLogitsProcessor:
    """Guard reasoning boundaries and Gemma4 tool-call structure."""

    _STATE_NORMAL = 0
    _STATE_TOOL = 1
    _STATE_POST_TOOL = 2

    def __init__(
        self,
        *,
        reasoning_open: bool,
        reasoning_start_token_ids: tuple[int, ...],
        reasoning_end_token_ids: tuple[int, ...],
        tool_call_start_token_id: int,
        tool_grammar: Any,
        eos_token_ids: tuple[int, ...],
        whitespace_token_ids: tuple[int, ...],
    ):
        self._reasoning_open = reasoning_open
        self._reasoning_open_mx = mx.array(reasoning_open)
        self._reasoning_start_first_token_id = reasoning_start_token_ids[0]
        self._reasoning_start_second_token_id = reasoning_start_token_ids[1]
        self._reasoning_end_token_id = reasoning_end_token_ids[0]
        self._tool_call_start_token_id = tool_call_start_token_id
        self._tool_grammar = tool_grammar
        self._initial_tool_token_ids = tuple(tool_grammar.initial_token_ids)
        self._post_tool_token_ids = (
            *eos_token_ids,
            *whitespace_token_ids,
            tool_call_start_token_id,
        )
        self._previous_token_mx = mx.array(0)
        self._context_token_count = 0
        self._reset_tool_state()

    def __call__(self, tokens: Any, logits: Any) -> Any:
        # VLM calls this once on prompt prefill. Prompt reasoning state already
        # came from decoded prompt text. Historical prompt tool calls are
        # ignored; the structural grammar starts on generated <|tool_call> only.
        token_ids = tokens.tolist()
        self._previous_token_mx = mx.array(token_ids[-1])
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

    def _reset_tool_state(self) -> None:
        self._tool_state = self._STATE_NORMAL
        self._tool_matcher = None

    def _start_tool_call(self) -> None:
        self._tool_state = self._STATE_TOOL
        self._tool_matcher = self._tool_grammar.start_matcher()

    def _process_tool_context_tokens(self, token_context: list[int]) -> None:
        for token_id in token_context[self._context_token_count :]:
            self._process_tool_context_token(token_id)
        self._context_token_count = len(token_context)

    def _process_tool_context_token(self, token_id: int) -> None:
        if self._tool_state == self._STATE_TOOL:
            self._consume_tool_token(token_id)
        elif token_id == self._tool_call_start_token_id:
            self._start_tool_call()

    def _consume_tool_token(self, token_id: int) -> None:
        complete = self._tool_grammar.consume_token(self._tool_matcher, token_id)
        if complete:
            self._tool_state = self._STATE_POST_TOOL
            self._tool_matcher = None

    def _process_last_token_mx(self, token_id: mx.array, logits: mx.array) -> mx.array:
        # Keep normal decode token handling in MLX: last_token.tolist() calls
        # eval()/wait and can create a per-token graph break. We only sync the
        # sampled token while an llguidance tool grammar is active, where slower
        # decoding is acceptable.
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
        # KISS structural scope: once <|tool_call> is emitted, llguidance
        # enforces one native Gemma4 call:
        #   call:KNOWN_TOOL{...}<tool_call|>
        # We still keep the cheap Python wrapper for lazy activation and the
        # post-tool state that allows only EOS/whitespace/another adjacent call.
        if self._tool_state == self._STATE_NORMAL:
            return _force_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._initial_tool_token_ids,
            )

        if self._tool_state == self._STATE_TOOL:
            self._consume_tool_token(int(token_id.item()))
            # The consumed sampled token will be appended to token_context after
            # this step. Skip it there so llguidance does not see it twice.
            self._context_token_count += 1
            if self._tool_state == self._STATE_POST_TOOL:
                return _force_token_ids_mx(
                    logits,
                    mx.array(True),
                    self._post_tool_token_ids,
                )
            return self._tool_grammar.mask_logits(self._tool_matcher, logits)

        if self._tool_state == self._STATE_POST_TOOL:
            logits = _force_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._initial_tool_token_ids,
            )
            return _force_token_ids_mx(
                logits,
                token_id != self._tool_call_start_token_id,
                self._post_tool_token_ids,
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
) -> Gemma4ReasoningGuardLogitsProcessor:
    tool_call_start_token_id = tokenizer.tool_call_start_tokens[0]
    tool_grammar = _Gemma4LLGuidanceToolGrammar(
        tokenizer=tokenizer,
        tool_names=context.tool_names,
        call_prefix_token_ids=_encode_token_ids(tokenizer, _GEMMA4_CALL_PREFIX),
        vocab_size=int(tokenizer.vocab_size),
    )

    return Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=context.reasoning_open,
        reasoning_start_token_ids=tokenizer.think_start_tokens,
        reasoning_end_token_ids=tokenizer.think_end_tokens,
        tool_call_start_token_id=tool_call_start_token_id,
        tool_grammar=tool_grammar,
        eos_token_ids=tuple(int(token_id) for token_id in tokenizer.eos_token_ids),
        whitespace_token_ids=tuple(
            _encode_token_ids(tokenizer, whitespace)[0]
            for whitespace in _GEMMA4_WHITESPACE
        ),
    )


def _gemma4_llguidance_grammar(tool_names: tuple[str, ...]) -> str:
    tool_choice = " | ".join(json.dumps(tool_name) for tool_name in tool_names)
    return rf'''%llguidance {{}}
start: "call:" tool object <tool_call|>
tool: {tool_choice}
object: "{{" WS (member (WS "," WS member)*)? WS "}}"
member: key WS ":" WS value
key: IDENT | gemma_string
IDENT: /[A-Za-z_]/ /[A-Za-z0-9_.$\/-]*/
value: gemma_string | object | array | NUMBER | "true" | "false" | "null"
array: "[" WS (value (WS "," WS value)*)? WS "]"
gemma_string: <|"|> STRING_CHAR* <|"|>
STRING_CHAR: /[^<]/ | "<" /[^|]/
NUMBER: "-"? (/[0-9]/ | /[1-9]/ /[0-9]*/) ("." /[0-9]/+)? (/[eE]/ /[+-]/? /[0-9]/+)?
WS: /[ \t\n\r]*/
'''


def _get_llguidance_tokenizer(tokenizer: Any) -> Any:
    hf_tokenizer = tokenizer._tokenizer
    cache_key = id(hf_tokenizer)
    llg_tokenizer = _LLG_TOKENIZER_CACHE.get(cache_key)
    if llg_tokenizer is None:
        llg_tokenizer = llguidance.hf.from_tokenizer(
            hf_tokenizer,
            n_vocab=int(tokenizer.vocab_size),
            eos_token=list(tokenizer.eos_token_ids),
        )
        _LLG_TOKENIZER_CACHE[cache_key] = llg_tokenizer
    return llg_tokenizer


def _force_token_ids_mx(
    logits: mx.array,
    condition: mx.array,
    token_ids: tuple[int, ...],
) -> mx.array:
    token_id_list = list(token_ids)
    logits[:, token_id_list] = mx.where(
        condition,
        _FORCED_TOOL_LOGIT,
        logits[:, token_id_list],
    )
    return logits


def _encode_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(token_id)
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    )
