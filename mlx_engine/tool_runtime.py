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
    QWEN35_FUNCTION_START,
    QWEN35_TOOL_CALL_END,
    QWEN35_TOOL_CALL_START,
    QWEN35_TOOLS_END,
    QWEN35_TOOLS_START,
    Gemma4ToolContext,
    Qwen35ToolContext,
    gemma4_reasoning_is_open,
    qwen35_reasoning_is_open,
)

# Match Gemma4 declaration blocks like:
#   <|tool>declaration:get_weather{ ... }<tool|>
# and capture only the declared tool name (`get_weather`).
_GEMMA4_TOOL_NAME_RE = re.compile(
    rf"{re.escape(GEMMA4_TOOL_DECLARATION_START)}.*?"
    r"declaration:\s*([A-Za-z_][A-Za-z0-9_.$/:-]*)\s*{.*?"
    rf"{re.escape(GEMMA4_TOOL_DECLARATION_END)}",
    re.DOTALL,
)
_QWEN35_TOOLS_RE = re.compile(
    rf"{re.escape(QWEN35_TOOLS_START)}(.*?){re.escape(QWEN35_TOOLS_END)}",
    re.DOTALL,
)
_GEMMA4_CALL_PREFIX = "call:"
_TOOL_WHITESPACE = (" ", "\n", "\t", "\r")
# For tiny forced-token states, update only the allowed token ids instead of
# adding an unnecessary full-vocabulary mask.
_FORCED_TOOL_LOGIT = 1e9
_LLG_TOKENIZER_CACHE: dict[tuple[int, int, tuple[int, ...]], Any] = {}


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


def create_qwen35_tool_context_from_prompt(
    *,
    tokenizer: Any,
    prompt_tokens: list[int],
    model_type: str | None,
) -> Qwen35ToolContext | None:
    """Return Qwen3.5 tool context only for rendered native tool prompts."""
    if not str(model_type or "").startswith("qwen3_5"):
        return None
    if tokenizer.tool_call_start != QWEN35_TOOL_CALL_START:
        return None
    if tokenizer.tool_call_end != QWEN35_TOOL_CALL_END:
        return None

    prompt_text = tokenizer.decode(prompt_tokens)
    if (
        QWEN35_TOOL_CALL_START not in prompt_text
        or QWEN35_FUNCTION_START not in prompt_text
    ):
        return None

    tool_names = _qwen35_tool_names_from_prompt(prompt_text)
    if len(tool_names) == 0:
        return None

    return Qwen35ToolContext(
        tool_names=tool_names,
        reasoning_open=qwen35_reasoning_is_open(prompt_text),
    )


def _qwen35_tool_names_from_prompt(prompt_text: str) -> tuple[str, ...]:
    tool_names: list[str] = []
    for tools_block in _QWEN35_TOOLS_RE.findall(prompt_text):
        for line in tools_block.splitlines():
            line = line.strip()
            if line == "":
                continue
            try:
                tool = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = tool.get("function", {}).get("name")
            if isinstance(name, str) and name != "":
                tool_names.append(name)
    return tuple(dict.fromkeys(tool_names))


class _LLGuidanceToolGrammar:
    """llguidance-backed grammar for one native tool call."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        grammar: str,
        initial_token_ids: tuple[int, ...],
    ):
        self.initial_token_ids = tuple(dict.fromkeys(initial_token_ids))
        self._tokenizer = tokenizer
        self._grammar = grammar
        self._llg_tokenizer = None
        self._llg_vocab_size = 0
        self._bitmask = None
        self._bitmask_vocab_size = 0

    def _ensure_ready(self, vocab_size: int) -> None:
        vocab_size = max(_tokenizer_vocab_size(self._tokenizer), int(vocab_size))
        if self._llg_tokenizer is not None and self._llg_vocab_size == vocab_size:
            return
        self._llg_tokenizer = _get_llguidance_tokenizer(self._tokenizer, vocab_size)
        self._llg_vocab_size = vocab_size
        self._bitmask = None
        self._bitmask_vocab_size = 0

    def start_matcher(self, vocab_size: int) -> Any:
        self._ensure_ready(vocab_size)
        return llguidance.LLMatcher(self._llg_tokenizer, self._grammar)

    def consume_token(self, matcher: Any, token_id: int) -> bool:
        matcher.consume_token(token_id)
        if error := matcher.get_error():
            raise ValueError(error)
        return matcher.is_stopped()

    def mask_logits(self, matcher: Any, logits: mx.array) -> mx.array:
        if self._bitmask is None or self._bitmask_vocab_size != self._llg_vocab_size:
            self._bitmask = llguidance.numpy.allocate_token_bitmask(
                1, self._llg_vocab_size
            )
            self._bitmask_vocab_size = self._llg_vocab_size
        llguidance.numpy.fill_next_token_bitmask(matcher, self._bitmask, 0)
        return llguidance.mlx.apply_token_bitmask(logits, self._bitmask)


class NativeToolReasoningGuardLogitsProcessor:
    """Guard reasoning boundaries and native tool-call structure."""

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
        """Initialize native marker ids, tool grammar, and reasoning state."""
        self._reasoning_open = reasoning_open
        self._reasoning_open_mx = mx.array(reasoning_open)
        self._reasoning_start_first_token_id = reasoning_start_token_ids[0]
        self._reasoning_start_second_token_id = (
            reasoning_start_token_ids[1]
            if len(reasoning_start_token_ids) == 2
            else None
        )
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
        """Initialize from VLM prefill tokens and apply prompt-time masks."""
        # VLM calls this once on prompt prefill. Prompt reasoning state already
        # came from decoded prompt text. Historical prompt tool calls are
        # ignored; the structural grammar starts on generated tool calls only.
        token_ids = tokens.tolist()
        self._previous_token_mx = mx.array(token_ids[-1])
        self._reasoning_open_mx = mx.array(self._reasoning_open)
        self._context_token_count = len(token_ids)
        self._reset_tool_state()

        if self._reasoning_open:
            # The prompt already ended inside visible reasoning, so prevent the
            # first generated token from starting a tool call before reasoning closes.
            logits[:, self._tool_call_start_token_id] = -float("inf")
        return logits

    def process_last_token_with_context(
        self,
        token_context: list[int],
        last_token: mx.array,
        logits: mx.array,
    ) -> mx.array:
        """Catch up from materialized context, then mask next-token logits."""
        for token_id in token_context[self._context_token_count :]:
            if self._tool_state == self._STATE_TOOL:
                # This token was already materialized by the batcher; feed it
                # into llguidance so the grammar is caught up before masking.
                self._consume_tool_grammar_token(token_id)
            elif token_id == self._tool_call_start_token_id:
                # A materialized tool-call marker starts grammar tracking. The
                # grammar itself begins after that protocol-specific marker.
                self._tool_state = self._STATE_TOOL
                self._tool_matcher = self._tool_grammar.start_matcher(
                    int(logits.shape[-1])
                )
        self._context_token_count = len(token_context)

        return self._process_last_token_mx(last_token.reshape(-1)[0], logits)

    def _reset_tool_state(self) -> None:
        """Return tool-grammar tracking to ordinary non-tool generation."""
        self._tool_state = self._STATE_NORMAL
        self._tool_matcher = None

    def _consume_tool_grammar_token(self, token_id: int) -> None:
        """Feed one generated tool token into llguidance and mark completion."""
        complete = self._tool_grammar.consume_token(self._tool_matcher, token_id)
        if complete:
            self._tool_state = self._STATE_POST_TOOL
            self._tool_matcher = None

    def _process_last_token_mx(self, token_id: mx.array, logits: mx.array) -> mx.array:
        """Track reasoning state in MLX and apply next-token masks."""
        # Keep normal decode token handling in MLX: last_token.tolist() calls
        # eval()/wait and can create a per-token graph break. We only sync the
        # sampled token while an llguidance tool grammar is active.

        if self._reasoning_start_second_token_id is None:
            # Qwen-style reasoning opens with a single <think> token.
            reasoning_start = token_id == self._reasoning_start_first_token_id
        else:
            # Gemma-style reasoning opens with a two-token marker.
            reasoning_start = (
                self._previous_token_mx == self._reasoning_start_first_token_id
            ) & (token_id == self._reasoning_start_second_token_id)

        # Visible reasoning closes with a single protocol-specific token.
        reasoning_end = token_id == self._reasoning_end_token_id

        # If the opening marker just completed, mark reasoning as open.
        self._reasoning_open_mx = mx.where(
            reasoning_start,
            mx.array(True),
            self._reasoning_open_mx,
        )

        # If the close marker was sampled, mark reasoning as closed.
        self._reasoning_open_mx = mx.where(
            reasoning_end,
            mx.array(False),
            self._reasoning_open_mx,
        )

        # Remember this token so the next step can detect two-token openers.
        self._previous_token_mx = token_id

        if self._tool_state == self._STATE_NORMAL:
            # Bridge the first token after a tool-call start without syncing
            # token_id to Python. Starting llguidance here would require
            # token_id.item() on every normal decode step; instead, use an MLX
            # condition to force the grammar's valid first token only when the
            # tool-call marker was sampled.
            logits = _boost_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._initial_tool_token_ids,
            )

        elif self._tool_state == self._STATE_TOOL:
            # We are inside a native tool call. This is the one place we sync
            # token_id to Python so llguidance can advance.
            self._consume_tool_grammar_token(int(token_id.item()))
            # The consumed sampled token will be appended to token_context after
            # this step. Skip it there so llguidance does not see it twice.
            self._context_token_count += 1
            if self._tool_state == self._STATE_POST_TOOL:
                # The call just closed. Allow only EOS, whitespace, or another
                # adjacent tool call, preserving the model's scores among them.
                logits = _mask_except_token_ids_mx(
                    logits,
                    mx.array(True),
                    self._post_tool_token_ids,
                )
            else:
                # The call is still open. Let llguidance choose the valid next
                # tokens for this protocol's native call grammar.
                logits = self._tool_grammar.mask_logits(self._tool_matcher, logits)

        elif self._tool_state == self._STATE_POST_TOOL:
            # If another tool call was sampled, force its first grammar token.
            logits = _boost_token_ids_mx(
                logits,
                token_id == self._tool_call_start_token_id,
                self._initial_tool_token_ids,
            )
            # Otherwise stay in the post-tool lane: EOS, whitespace, or another
            # adjacent tool call. Preserve scores among those continuations.
            logits = _mask_except_token_ids_mx(
                logits,
                token_id != self._tool_call_start_token_id,
                self._post_tool_token_ids,
            )

        # Keep tool-call starts blocked while visible reasoning is open. This
        # forces the model to sample the real reasoning close marker first.
        logits[:, self._tool_call_start_token_id] = mx.where(
            self._reasoning_open_mx,
            -float("inf"),
            logits[:, self._tool_call_start_token_id],
        )
        return logits


class Gemma4ReasoningGuardLogitsProcessor(NativeToolReasoningGuardLogitsProcessor):
    """Guard reasoning boundaries and Gemma4 tool-call structure."""


class Qwen35ReasoningGuardLogitsProcessor(NativeToolReasoningGuardLogitsProcessor):
    """Guard reasoning boundaries and Qwen3.5 tool-call structure."""


def create_gemma4_reasoning_guard_logits_processor(
    *,
    tokenizer: Any,
    context: Gemma4ToolContext,
) -> Gemma4ReasoningGuardLogitsProcessor:
    tool_call_start_token_id = tokenizer.tool_call_start_tokens[0]
    tool_grammar = _LLGuidanceToolGrammar(
        tokenizer=tokenizer,
        grammar=_gemma4_llguidance_grammar(context.tool_names),
        initial_token_ids=(_encode_token_ids(tokenizer, _GEMMA4_CALL_PREFIX)[0],),
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
            for whitespace in _TOOL_WHITESPACE
        ),
    )


def create_qwen35_reasoning_guard_logits_processor(
    *,
    tokenizer: Any,
    context: Qwen35ToolContext,
) -> Qwen35ReasoningGuardLogitsProcessor:
    tool_call_start_token_id = tokenizer.tool_call_start_tokens[0]
    tool_grammar = _LLGuidanceToolGrammar(
        tokenizer=tokenizer,
        grammar=_qwen35_llguidance_grammar(
            context.tool_names,
            tool_call_end_token_id=tokenizer.tool_call_end_tokens[0],
        ),
        initial_token_ids=(_encode_token_ids(tokenizer, "\n")[0],),
    )

    return Qwen35ReasoningGuardLogitsProcessor(
        reasoning_open=context.reasoning_open,
        reasoning_start_token_ids=tokenizer.think_start_tokens,
        reasoning_end_token_ids=tokenizer.think_end_tokens,
        tool_call_start_token_id=tool_call_start_token_id,
        tool_grammar=tool_grammar,
        eos_token_ids=tuple(int(token_id) for token_id in tokenizer.eos_token_ids),
        whitespace_token_ids=tuple(
            _encode_token_ids(tokenizer, whitespace)[0]
            for whitespace in _TOOL_WHITESPACE
        ),
    )


def _gemma4_llguidance_grammar(tool_names: tuple[str, ...]) -> str:
    tool_choice = " | ".join(json.dumps(tool_name) for tool_name in tool_names)
    return rf"""%llguidance {{}}
start: "call:" tool WS object WS <tool_call|>
tool: {tool_choice}
object: "{{" WS (member (WS "," WS member)*)? WS "}}"
member: key WS ":" WS value
key: BARE_KEY | gemma_string
BARE_KEY: /[A-Za-z0-9_.$\/-]/+
value: gemma_string | object | array | NUMBER | LITERAL
LITERAL: "true" | "false" | "null" | "None" | "none"
array: "[" WS (value (WS "," WS value)*)? WS "]"
gemma_string: <|"|> STRING_CHAR* <|"|>
STRING_CHAR: /[^<]/ | "<" /[^|]/ | "<|" /[^"]/ | "<|\"" /[^|]/ | "<|\"|" /[^>]/
NUMBER: "-"? (/[0-9]/ | /[1-9]/ /[0-9]*/) ("." /[0-9]/+)? (/[eE]/ /[+-]/? /[0-9]/+)?
WS: /[ \t\n\r]*/
"""


def _qwen35_llguidance_grammar(
    tool_names: tuple[str, ...],
    *,
    tool_call_end_token_id: int,
) -> str:
    tool_choice = " | ".join(json.dumps(tool_name) for tool_name in tool_names)

    return rf"""%llguidance {{}}
start: WS "<function=" WS tool ">" WS parameter* "</function>" WS <[{int(tool_call_end_token_id)}]>
tool: {tool_choice}
parameter: "<parameter=" PARAM_NAME ">" param_value "</parameter>" WS
PARAM_NAME: /[^>]/+
param_value: PARAM_CHAR*
PARAM_CHAR: /[^<]/ | "<" /[^\/]/ | "</" /[^p]/ | "</p" /[^a]/ | "</pa" /[^r]/ | "</par" /[^a]/ | "</para" /[^m]/ | "</param" /[^e]/ | "</parame" /[^t]/ | "</paramet" /[^e]/ | "</paramete" /[^r]/ | "</parameter" /[^>]/
WS: /[ \t\n\r]*/
"""


def _get_llguidance_tokenizer(tokenizer: Any, vocab_size: int) -> Any:
    hf_tokenizer = tokenizer._tokenizer
    eos_token_ids = tuple(sorted(int(token_id) for token_id in tokenizer.eos_token_ids))
    cache_key = (id(hf_tokenizer), int(vocab_size), eos_token_ids)
    llg_tokenizer = _LLG_TOKENIZER_CACHE.get(cache_key)
    if llg_tokenizer is None:
        llg_tokenizer = llguidance.hf.from_tokenizer(
            hf_tokenizer,
            n_vocab=int(vocab_size),
            eos_token=list(eos_token_ids),
        )
        _LLG_TOKENIZER_CACHE[cache_key] = llg_tokenizer
    return llg_tokenizer


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab()
    return max(
        int(tokenizer.vocab_size), max(int(token_id) for token_id in vocab.values()) + 1
    )


def _boost_token_ids_mx(
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


def _mask_except_token_ids_mx(
    logits: mx.array,
    condition: mx.array,
    token_ids: tuple[int, ...],
) -> mx.array:
    allowed_logits = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    token_id_list = list(token_ids)
    allowed_logits[:, token_id_list] = logits[:, token_id_list]
    return mx.where(condition, allowed_logits, logits)


def _encode_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)
    )
