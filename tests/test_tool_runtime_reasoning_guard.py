import mlx.core as mx

from mlx_engine.tool_protocols import (
    GEMMA4_CHANNEL_END,
    GEMMA4_REASONING_START,
    GEMMA4_TOOL_CALL_START,
    Gemma4ToolContext,
    Qwen35ToolContext,
)
from mlx_engine.tool_runtime import (
    Gemma4ReasoningGuardLogitsProcessor,
    Qwen35ReasoningGuardLogitsProcessor,
    _LLGuidanceToolGrammar,
    _gemma4_llguidance_grammar,
    _qwen35_llguidance_grammar,
    _tokenizer_vocab_size,
    create_gemma4_reasoning_guard_logits_processor,
    create_qwen35_reasoning_guard_logits_processor,
)


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
            19: "\t",
            20: "\r",
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


class _QwenTokenizer:
    def __init__(self):
        self.decode_count = 0
        self.vocab_size = 24
        self.eos_token_ids = {0}
        self.think_start_tokens = (1,)
        self.think_end_tokens = (2,)
        self.tool_call_start_tokens = (4,)
        self.tool_call_end_tokens = (5,)
        self.text_by_token_id = {
            0: "<eos>",
            1: "<think>",
            2: "</think>",
            4: "<tool_call>",
            5: "</tool_call>",
            6: "<",
            7: "function",
            8: "=",
            15: " ",
            17: "\n",
            19: "\t",
            20: "\r",
        }
        self.token_id_by_text = {
            text: token_id for token_id, text in self.text_by_token_id.items()
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<function=":
            return [6, 7, 8]
        if text in self.token_id_by_text:
            return [self.token_id_by_text[text]]
        return []

    def decode(self, token_ids):
        self.decode_count += 1
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.text_by_token_id[int(token_id)] for token_id in token_ids)

    def get_added_vocab(self):
        return {
            "<tool_call>": 4,
            "</tool_call>": 5,
            "<think>": 1,
            "</think>": 2,
        }

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


class _FakeToolGrammar:
    initial_token_ids = (6,)

    def __init__(self, tool_names=("get_weather",)):
        name_tokens = {
            "get_weather": (8, 9, 10),
            "search": (16,),
        }
        self._headers = [
            (6, 7, *name_tokens[tool_name], 11) for tool_name in tool_names
        ]

    def start_matcher(self, *_args):
        return {
            "state": "header",
            "pos": 0,
            "headers": self._headers,
            "depth": 0,
            "in_string": False,
        }

    def consume_token(self, matcher, token_id: int) -> bool:
        if matcher["state"] == "header":
            pos = matcher["pos"]
            headers = [
                header for header in matcher["headers"] if header[pos] == token_id
            ]
            if not headers:
                raise ValueError(f"unexpected header token {token_id}")
            matcher["headers"] = headers
            matcher["pos"] = pos + 1
            if any(matcher["pos"] == len(header) for header in headers):
                matcher["state"] = "args"
                matcher["depth"] = 1
            return False

        if matcher["state"] == "args":
            if token_id == 13:
                matcher["in_string"] = not matcher["in_string"]
                return False
            if matcher["in_string"]:
                return False
            if token_id == 11:
                matcher["depth"] += 1
            elif token_id == 12:
                matcher["depth"] -= 1
                if matcher["depth"] == 0:
                    matcher["state"] = "need_end"
            return False

        if matcher["state"] == "need_end":
            if token_id != 5:
                raise ValueError(f"expected tool-call end, got {token_id}")
            matcher["state"] = "post"
            return True

        raise ValueError(f"unexpected token after complete tool call {token_id}")

    def mask_logits(self, matcher, logits):
        if matcher["state"] == "header":
            pos = matcher["pos"]
            allowed = sorted({header[pos] for header in matcher["headers"]})
            logits[:, allowed] = 1e9
        elif matcher["state"] == "args":
            logits[:, [0, 4, 5]] = -float("inf")
        elif matcher["state"] == "need_end":
            logits[:, [5]] = 1e9
        return logits


def _context(tool_names=("get_weather",), reasoning_open=False):
    return Gemma4ToolContext(tool_names=tool_names, reasoning_open=reasoning_open)


def _processor(tool_names=("get_weather",), reasoning_open=False):
    processor = Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=reasoning_open,
        reasoning_start_token_ids=(1, 2),
        reasoning_end_token_ids=(3,),
        tool_call_start_token_id=4,
        tool_grammar=_FakeToolGrammar(tool_names),
        eos_token_ids=(0,),
        whitespace_token_ids=(15, 17),
    )
    processor(mx.array([14], dtype=mx.int32), mx.zeros((1, 24), dtype=mx.float32))
    return processor


def _qwen_context(tool_names=("get_weather",), reasoning_open=False):
    return Qwen35ToolContext(tool_names=tool_names, reasoning_open=reasoning_open)


def _qwen_processor(tool_names=("get_weather",), reasoning_open=False):
    processor = Qwen35ReasoningGuardLogitsProcessor(
        reasoning_open=reasoning_open,
        reasoning_start_token_ids=(1,),
        reasoning_end_token_ids=(2,),
        tool_call_start_token_id=4,
        tool_grammar=_FakeToolGrammar(tool_names),
        eos_token_ids=(0,),
        whitespace_token_ids=(15, 17),
    )
    processor(mx.array([14], dtype=mx.int32), mx.zeros((1, 24), dtype=mx.float32))
    return processor


def test_gemma4_reasoning_guard_masks_tool_call_start_when_prompt_reasoning_open():
    processor = _processor(reasoning_open=True)
    logits = _FakeLogits(vocab_size=8)

    assert processor(mx.array([1, 2], dtype=mx.int32), logits) is logits

    assert logits.values[0][4] == -float("inf")


def test_gemma4_reasoning_guard_tracks_generated_reasoning_markers():
    processor = _processor()
    context = _mx_context()

    _process_token(processor, context, 1)
    logits = _process_token(processor, context, 2)

    assert logits[:, 4].tolist() == [-float("inf")]


def test_gemma4_reasoning_guard_allows_tool_call_start_after_channel_end():
    processor = _processor(reasoning_open=True)
    context = _mx_context()

    logits = _process_token(processor, context, 3)

    assert logits[:, 4].tolist() == [0.0]


def test_gemma4_reasoning_guard_does_not_decode_during_generation():
    tokenizer = _Tokenizer()
    processor = create_gemma4_reasoning_guard_logits_processor(
        tokenizer=tokenizer,
        context=_context(reasoning_open=True),
    )
    decode_count_after_setup = tokenizer.decode_count
    processor(mx.array([1, 2], dtype=mx.int32), mx.zeros((1, 24), dtype=mx.float32))
    context = [1, 2]
    _process_token(processor, context, 3)
    _process_token(processor, context, 1)
    _process_token(processor, context, 2)

    assert tokenizer.decode_count == decode_count_after_setup


def test_qwen35_reasoning_guard_tracks_single_token_reasoning_markers():
    processor = _qwen_processor()
    context = _mx_context()

    logits = _process_token(processor, context, 1)
    assert logits[:, 4].tolist() == [-float("inf")]

    logits = _process_token(processor, context, 2)
    assert logits[:, 4].tolist() == [0.0]


def test_qwen35_reasoning_guard_does_not_decode_during_generation():
    tokenizer = _QwenTokenizer()
    processor = create_qwen35_reasoning_guard_logits_processor(
        tokenizer=tokenizer,
        context=_qwen_context(reasoning_open=True),
    )
    decode_count_after_setup = tokenizer.decode_count
    processor(mx.array([1], dtype=mx.int32), mx.zeros((1, 24), dtype=mx.float32))
    context = [1]
    _process_token(processor, context, 2)
    _process_token(processor, context, 1)

    assert tokenizer.decode_count == decode_count_after_setup


def test_qwen35_structure_bridges_after_tool_call_start():
    processor = _qwen_processor()
    context = _mx_context()

    logits = _process_token(processor, context, 4)

    assert _forced_token_ids(logits) == [6]


def _mx_context():
    return [14]


def _process_token(processor, context, token_id, logits=None):
    logits = processor.process_last_token_with_context(
        context,
        mx.array([token_id], dtype=mx.int32),
        _mx_logits() if logits is None else logits,
    )
    context.append(token_id)
    return logits


def _mx_logits():
    return mx.zeros((1, 24), dtype=mx.float32)


def _post_tool_logits():
    values = [-10.0] * 24
    values[0] = 1.0
    values[4] = 3.0
    values[15] = 2.0
    values[17] = 0.5
    values[14] = 100.0
    return mx.array([values], dtype=mx.float32)


def _assert_post_tool_logits_preserved(logits):
    assert logits[:, 0].tolist() == [1.0]
    assert logits[:, 4].tolist() == [3.0]
    assert logits[:, 15].tolist() == [2.0]
    assert logits[:, 17].tolist() == [0.5]
    assert logits[:, 14].tolist() == [-float("inf")]
    assert mx.argmax(logits, axis=-1).tolist() == [4]


def _forced_token_ids(logits):
    return [
        token_id
        for token_id, value in enumerate(logits.reshape(-1).tolist())
        if value > 1e8
    ]


def test_gemma4_structure_constrains_header_after_tool_call_start():
    processor = _processor()
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
    processor = _processor(tool_names=("get_weather", "search"))
    context = _mx_context()
    _process_token(processor, context, 4)
    _process_token(processor, context, 6)

    logits = _process_token(processor, context, 7)

    assert _forced_token_ids(logits) == [8, 16]


def test_gemma4_structure_balances_args_and_requires_tool_end_marker():
    processor = _processor()
    context = _mx_context()
    for token_id in [4, 6, 7, 8, 9, 10]:
        _process_token(processor, context, token_id)

    logits = _process_token(processor, context, 11)
    assert logits[:, 5].tolist() == [-float("inf")]
    assert logits[:, 12].tolist() == [0.0]

    logits = _process_token(processor, context, 12)
    assert _forced_token_ids(logits) == [5]

    logits = _process_token(processor, context, 5)
    assert logits[:, 0].tolist() == [0.0]
    assert logits[:, 4].tolist() == [0.0]
    assert logits[:, 15].tolist() == [0.0]
    assert logits[:, 14].tolist() == [-float("inf")]


def test_gemma4_post_tool_mask_preserves_allowed_logits():
    processor = _processor()
    context = _mx_context()
    for token_id in [4, 6, 7, 8, 9, 10, 11, 12]:
        _process_token(processor, context, token_id)

    _assert_post_tool_logits_preserved(
        _process_token(processor, context, 5, _post_tool_logits())
    )

    _assert_post_tool_logits_preserved(
        _process_token(processor, context, 15, _post_tool_logits())
    )


def test_gemma4_structure_ignores_braces_inside_gemma_strings():
    processor = _processor()
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


def test_gemma4_llguidance_grammar_uses_native_special_tokens():
    grammar = _gemma4_llguidance_grammar(("get_weather", "search-tool"))

    assert '"get_weather"' in grammar
    assert '"search-tool"' in grammar
    assert "<tool_call|>" in grammar
    assert '<|"|>' in grammar


def test_gemma4_llguidance_grammar_accepts_parser_edge_cases():
    import llguidance
    import llguidance.hf
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    special_tokens = [
        "<eos>",
        '<|"|>',
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|image>",
        "<image|>",
        "<|image|>",
        "<|audio>",
        "<audio|>",
        "<|audio|>",
        "<|video|>",
    ]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [
            'call:lookup{2fa_code:<|"|>what is <|tool_call>?<|"|>}<tool_call|>',
            "call:lookup{optional:None}<tool_call|>",
            "call:lookup{optional:none}<tool_call|>",
        ],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", *special_tokens],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=special_tokens[1:],
    )
    llg_tokenizer = llguidance.hf.from_tokenizer(
        hf_tokenizer,
        n_vocab=len(hf_tokenizer.get_vocab()),
        eos_token=[hf_tokenizer.eos_token_id],
    )
    grammar = _gemma4_llguidance_grammar(("lookup",))

    for text in [
        'call:lookup{2fa_code:<|"|>what is <|tool_call>?<|"|>}<tool_call|>',
        "call:lookup{optional:None}<tool_call|>",
        "call:lookup{optional:none}<tool_call|>",
    ]:
        matcher = llguidance.LLMatcher(llg_tokenizer, grammar)
        for token_id in hf_tokenizer.encode(text, add_special_tokens=False):
            matcher.consume_token(token_id)
            assert not matcher.get_error()
        assert matcher.is_stopped()


def _qwen35_hf_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [
            "<function=lookup><parameter=query>weather</parameter></function>",
            "<function=lookup><parameter=html><div>{\"a\":[1,2]}</div></parameter></function>",
        ],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    hf_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<tool_call>",
                "</tool_call>",
                "<think>",
                "</think>",
            ]
        }
    )
    return hf_tokenizer


def test_qwen35_llguidance_grammar_accepts_parser_edge_cases():
    import llguidance
    import llguidance.hf

    hf_tokenizer = _qwen35_hf_tokenizer()
    tool_call_end_token_id = hf_tokenizer.convert_tokens_to_ids("</tool_call>")
    value_special_token_ids = tuple(
        sorted(
            int(token_id)
            for token_text, token_id in hf_tokenizer.get_added_vocab().items()
            if "<" in token_text
        )
    )
    llg_tokenizer = llguidance.hf.from_tokenizer(
        hf_tokenizer,
        n_vocab=max(hf_tokenizer.get_vocab().values()) + 1,
        eos_token=[hf_tokenizer.eos_token_id],
    )
    grammar = _qwen35_llguidance_grammar(
        ("lookup", "search-tool"),
        tool_call_end_token_id=tool_call_end_token_id,
        value_special_token_ids=value_special_token_ids,
    )

    for text in [
        "\n<function=lookup>\n<parameter=2fa_code>\n123\n</parameter>\n</function>\n</tool_call>",
        "<function=lookup></function></tool_call>",
        "<function=lookup><parameter=query>what is <tool_call>?</parameter></function></tool_call>",
        "<function=lookup><parameter=html><div>{\"a\":[1,2]}</div>\n</parameter></function></tool_call>",
    ]:
        matcher = llguidance.LLMatcher(llg_tokenizer, grammar)
        for token_id in hf_tokenizer.encode(text, add_special_tokens=False):
            matcher.consume_token(token_id)
            assert not matcher.get_error()
        assert matcher.is_stopped()


def test_qwen35_llguidance_grammar_rejects_unknown_tool_names():
    import llguidance
    import llguidance.hf

    hf_tokenizer = _qwen35_hf_tokenizer()
    llg_tokenizer = llguidance.hf.from_tokenizer(
        hf_tokenizer,
        n_vocab=max(hf_tokenizer.get_vocab().values()) + 1,
        eos_token=[hf_tokenizer.eos_token_id],
    )
    grammar = _qwen35_llguidance_grammar(
        ("lookup",),
        tool_call_end_token_id=hf_tokenizer.convert_tokens_to_ids("</tool_call>"),
        value_special_token_ids=(),
    )
    matcher = llguidance.LLMatcher(llg_tokenizer, grammar)

    for token_id in hf_tokenizer.encode(
        "<function=unknown></function></tool_call>", add_special_tokens=False
    ):
        matcher.consume_token(token_id)
        if matcher.get_error():
            break

    assert matcher.get_error()


def test_llguidance_vocab_size_includes_added_tokens():
    import llguidance.hf
    from types import SimpleNamespace

    hf_tokenizer = _qwen35_hf_tokenizer()
    wrapper = SimpleNamespace(
        vocab_size=hf_tokenizer.vocab_size,
        get_vocab=hf_tokenizer.get_vocab,
    )
    vocab_size = _tokenizer_vocab_size(wrapper)

    assert vocab_size > hf_tokenizer.vocab_size
    llguidance.hf.from_tokenizer(
        hf_tokenizer,
        n_vocab=vocab_size,
        eos_token=[hf_tokenizer.eos_token_id],
    )


def test_llguidance_tool_grammar_sizes_bitmask_from_logits_vocab():
    from types import SimpleNamespace

    hf_tokenizer = _qwen35_hf_tokenizer()
    tokenizer = SimpleNamespace(
        _tokenizer=hf_tokenizer,
        vocab_size=hf_tokenizer.vocab_size,
        eos_token_ids={hf_tokenizer.eos_token_id},
        get_vocab=hf_tokenizer.get_vocab,
    )
    logits_vocab_size = max(hf_tokenizer.get_vocab().values()) + 50
    grammar = _LLGuidanceToolGrammar(
        tokenizer=tokenizer,
        grammar='%llguidance {}\nstart: "a"',
        initial_token_ids=(hf_tokenizer.encode("a", add_special_tokens=False)[0],),
    )

    matcher = grammar.start_matcher(logits_vocab_size)
    logits = grammar.mask_logits(
        matcher,
        mx.zeros((1, logits_vocab_size), dtype=mx.float32),
    )

    mx.eval(logits)
