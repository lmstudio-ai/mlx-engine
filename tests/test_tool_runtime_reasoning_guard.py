from mlx_engine.tool_protocols import (
    GEMMA4_CHANNEL_END,
    GEMMA4_REASONING_START,
    GEMMA4_TOOL_CALL_START,
    Gemma4ToolContext,
)
from mlx_engine.tool_runtime import create_gemma4_reasoning_guard_logits_processor


class _Tokenizer:
    def __init__(self, tool_call_start_tokens=(4,)):
        self.decode_count = 0
        self.think_start_tokens = (1,)
        self.think_end_tokens = (3,)
        self.tool_call_start_tokens = tool_call_start_tokens
        self.text_by_token_id = {
            1: GEMMA4_REASONING_START,
            2: "\nNeed data.",
            3: GEMMA4_CHANNEL_END,
            4: GEMMA4_TOOL_CALL_START,
            5: "ignored",
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == GEMMA4_REASONING_START:
            return list(self.think_start_tokens)
        if text == GEMMA4_CHANNEL_END:
            return list(self.think_end_tokens)
        if text == GEMMA4_TOOL_CALL_START:
            return list(self.tool_call_start_tokens)
        return []

    def decode(self, token_ids):
        self.decode_count += 1
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.text_by_token_id[int(token_id)] for token_id in token_ids)


class _FakeLogits:
    def __init__(self, vocab_size: int):
        self.shape = (1, vocab_size)
        self.values = [[0.0 for _ in range(vocab_size)]]

    def __setitem__(self, key, value):
        row_selector, token_id = key
        assert row_selector == slice(None)
        for row in self.values:
            row[token_id] = value


def _context(tool_names=("get_weather",), reasoning_open=False):
    return Gemma4ToolContext(tool_names=tool_names, reasoning_open=reasoning_open)


def test_gemma4_reasoning_guard_masks_tool_call_start_when_prompt_reasoning_open():
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(),
        context=_context(reasoning_open=True),
    )
    logits = _FakeLogits(vocab_size=8)

    assert processor is not None
    assert processor([1, 2], logits) is logits

    assert logits.values[0][4] == -float("inf")


def test_gemma4_reasoning_guard_tracks_generated_reasoning_markers():
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(),
        context=_context(),
    )
    logits = _FakeLogits(vocab_size=8)

    assert processor is not None
    processor([5], logits)
    processor.process_last_token([1], logits)

    assert logits.values[0][4] == -float("inf")


def test_gemma4_reasoning_guard_allows_tool_call_start_after_channel_end():
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(),
        context=_context(reasoning_open=True),
    )
    logits = _FakeLogits(vocab_size=8)

    assert processor is not None
    processor([1, 2], logits)
    logits_after_channel_end = _FakeLogits(vocab_size=8)
    processor.process_last_token([3], logits_after_channel_end)

    assert logits_after_channel_end.values[0][4] == 0.0


def test_gemma4_reasoning_guard_does_not_decode_during_generation():
    tokenizer = _Tokenizer()
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=tokenizer,
        context=_context(reasoning_open=True),
    )
    logits = _FakeLogits(vocab_size=8)

    assert processor is not None
    decode_count_after_setup = tokenizer.decode_count
    processor([1, 2], logits)
    processor.process_last_token([3], logits)
    processor.process_last_token([1], logits)

    assert tokenizer.decode_count == decode_count_after_setup


def test_gemma4_reasoning_guard_requires_declared_tools():
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(),
        context=_context(tool_names=()),
    )

    assert processor is None


def test_gemma4_reasoning_guard_requires_single_token_tool_call_start():
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(tool_call_start_tokens=(4, 5)),
        context=_context(),
    )

    assert processor is None
