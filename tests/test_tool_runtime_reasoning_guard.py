import mlx.core as mx

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
        self.vocab_size = 24
        self.eos_token_ids = {0}
        self.think_start_tokens = (1, 2)
        self.think_end_tokens = (3,)
        self.tool_call_start_tokens = tool_call_start_tokens
        self.tool_call_end_tokens = (5,)
        self.text_by_token_id = {
            0: "<eos>",
            1: "<|channel>",
            2: "thought",
            3: GEMMA4_CHANNEL_END,
            4: GEMMA4_TOOL_CALL_START,
            5: "<tool_call|>",
            6: "call",
            7: ":",
            8: "get",
            9: "_",
            10: "weather",
            11: "{",
            12: "}",
            13: '<|"|>',
            14: "ignored",
            15: " ",
            16: "search",
            17: "\n",
            18: "}}",
        }
        self.token_id_by_text = {
            text: token_id for token_id, text in self.text_by_token_id.items()
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == GEMMA4_REASONING_START:
            return list(self.think_start_tokens)
        if text == GEMMA4_CHANNEL_END:
            return list(self.think_end_tokens)
        if text == GEMMA4_TOOL_CALL_START:
            return list(self.tool_call_start_tokens)
        if text == "<tool_call|>":
            return list(self.tool_call_end_tokens)
        if text == "call:":
            return [6, 7]
        if text == "get_weather":
            return [8, 9, 10]
        if text == "search":
            return [16]
        if text in self.token_id_by_text:
            return [self.token_id_by_text[text]]
        return []

    def decode(self, token_ids):
        self.decode_count += 1
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.text_by_token_id[int(token_id)] for token_id in token_ids)

    def get_vocab(self):
        return dict(self.token_id_by_text)


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
    processor.process_last_token([2], logits)

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
    processor.process_last_token([2], logits)

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



def _mx_processor(tool_names=("get_weather",)):
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=_Tokenizer(),
        context=_context(tool_names=tool_names),
    )
    assert processor is not None
    processor([14], mx.zeros((1, 24), dtype=mx.float32))
    return processor


def _mx_context():
    return [14]


def _process_token(processor, context, token_id):
    logits = processor.process_last_token_with_context(
        context,
        mx.array([token_id], dtype=mx.int32),
        _mx_logits(),
    )
    context.append(token_id)
    return logits


def _mx_logits():
    return mx.zeros((1, 24), dtype=mx.float32)


def _forced_token_ids(logits):
    return [
        token_id
        for token_id, value in enumerate(logits.reshape(-1).tolist())
        if value > 1e8
    ]


def test_gemma4_structure_constrains_header_after_tool_call_start():
    processor = _mx_processor()

    context = _mx_context()

    logits = _process_token(processor, context, 4)
    assert _forced_token_ids(logits) == [6]

    logits = _process_token(processor, context, 6)
    assert _forced_token_ids(logits) == [7]

    logits = _process_token(processor, context, 7)
    assert _forced_token_ids(logits) == [8]

    logits = _process_token(processor, context, 8)
    assert _forced_token_ids(logits) == [9]

    logits = _process_token(processor, context, 9)
    assert _forced_token_ids(logits) == [10]

    logits = _process_token(processor, context, 10)
    assert _forced_token_ids(logits) == [11]


def test_gemma4_structure_allows_only_known_tool_name_prefixes():
    processor = _mx_processor(tool_names=("get_weather", "search"))
    context = _mx_context()
    _process_token(processor, context, 4)
    _process_token(processor, context, 6)

    logits = _process_token(processor, context, 7)

    assert _forced_token_ids(logits) == [8, 16]


def test_gemma4_structure_balances_args_and_requires_tool_end_marker():
    processor = _mx_processor()
    context = _mx_context()
    for token_id in [4, 6, 7, 8, 9, 10]:
        _process_token(processor, context, token_id)

    logits = _process_token(processor, context, 11)
    assert logits[:, 5].tolist() == [-float("inf")]
    assert logits[:, 12].tolist() == [0.0]

    logits = _process_token(processor, context, 12)
    assert _forced_token_ids(logits) == [5]

    logits = _process_token(processor, context, 5)
    assert 0 in _forced_token_ids(logits)
    assert 4 in _forced_token_ids(logits)
    assert 15 in _forced_token_ids(logits)
    assert 14 not in _forced_token_ids(logits)


def test_gemma4_structure_ignores_braces_inside_gemma_strings():
    processor = _mx_processor()
    context = _mx_context()
    for token_id in [4, 6, 7, 8, 9, 10, 11]:
        _process_token(processor, context, token_id)

    _process_token(processor, context, 13)
    logits = _process_token(processor, context, 12)
    assert logits[:, 5].tolist() == [-float("inf")]
    assert logits[:, 14].tolist() == [0.0]

    _process_token(processor, context, 13)
    logits = _process_token(processor, context, 12)
    assert _forced_token_ids(logits) == [5]
