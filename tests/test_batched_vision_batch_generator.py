import contextlib
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from mlx_engine.model_kit.batched_vision import batch_generator as batcher
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator,
    GenerationBatch,
    _PrefixCacheSaveState,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.tool_runtime import Gemma4ReasoningGuardLogitsProcessor


def _argmax_sampler(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


def _logits(batch_size: int, seq_len: int, vocab_size: int = 8):
    return mx.zeros((batch_size, seq_len, vocab_size), dtype=mx.float32)


def _bump(logits, token: int):
    bump = [0.0] * logits.shape[-1]
    bump[token] = 100.0
    return logits + mx.array([bump], dtype=mx.float32)


def _prefix_cache_save_states(count: int):
    return [_PrefixCacheSaveState([], 0, [], None) for _ in range(count)]


class _HistoryProcessor:
    def __init__(self, token: int):
        self.token = token
        self.calls = []

    def __call__(self, tokens, logits):
        self.calls.append(tokens.tolist())
        return _bump(logits, self.token)


class _IntLastTokenProcessor:
    def __init__(self, token: int):
        self.token = token
        self.calls = []
        self.last_token_calls = []

    def process_last_token(self, last_token, logits):
        if not isinstance(last_token, int):
            raise TypeError("last_token must be an int")
        self.last_token_calls.append(last_token)
        return _bump(logits, self.token)

    def __call__(self, tokens, logits):
        self.calls.append(tokens.tolist())
        return _bump(logits, self.token)


class _NoopToolGrammar:
    initial_token_ids = (7,)

    def start_matcher(self, *_args):
        raise AssertionError("tool grammar should not activate in this test")


class _FakeBatchCache:
    keys = True

    def __init__(self, name: str = "cache"):
        self.name = name
        self.state = mx.array([0], dtype=mx.int32)
        self.extended = []
        self.filtered = []
        self.extracted = []

    def extract(self, idx: int):
        self.extracted.append(idx)
        return _FakeScalarCache(f"{self.name}:{idx}")

    def extend(self, other):
        self.extended.append(other)

    def filter(self, keep):
        self.filtered.append(keep.tolist())


class ArraysCache(_FakeBatchCache):
    pass


class _FakeScalarCache:
    def __init__(self, name: str = "scalar"):
        self.name = name
        self.state = mx.array([0], dtype=mx.int32)
        self.merge_calls = []

    def merge(self, caches):
        self.merge_calls.append(caches)
        return _FakeBatchCache(f"merged:{self.name}")


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.model_type = None
        self.config = SimpleNamespace(use_bidirectional_attention=None)

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.tolist(),
                "inputs_embeds_shape": (
                    None if inputs_embeds is None else inputs_embeds.shape
                ),
                "n_to_process": kwargs.get("n_to_process"),
                "logits_to_keep": kwargs.get("logits_to_keep"),
                "position_ids": (
                    None
                    if kwargs.get("position_ids") is None
                    else kwargs["position_ids"].tolist()
                ),
                "rope_deltas": (
                    None
                    if kwargs.get("rope_deltas") is None
                    else kwargs["rope_deltas"].tolist()
                ),
                "mm_token_type_ids": (
                    None
                    if kwargs.get("mm_token_type_ids") is None
                    else kwargs["mm_token_type_ids"].tolist()
                ),
                "token_type_ids": (
                    None
                    if kwargs.get("token_type_ids") is None
                    else kwargs["token_type_ids"].tolist()
                ),
            }
        )
        batch_size, seq_len = input_ids.shape
        return SimpleNamespace(logits=_logits(batch_size, seq_len))


def _gemma4_unified_model():
    model = _FakeModel()
    model.model_type = "gemma4_unified"
    model.config = SimpleNamespace(use_bidirectional_attention="vision")
    return model


def _gemma4_model():
    model = _FakeModel()
    model.model_type = "gemma4"
    model.config = SimpleNamespace(use_bidirectional_attention="vision")
    return model


def _gemma4_non_bidir_model():
    model = _FakeModel()
    model.model_type = "gemma4"
    model.config = SimpleNamespace(use_bidirectional_attention=None)
    return model


def test_batch_generator_uses_vlm_prompt_cache_factory():
    model = _FakeModel()
    model.layers = [object()]

    prompt_cache = batcher.make_prompt_cache(model)

    assert type(prompt_cache[0]) is KVCache


def test_prefill_and_decode_honor_model_logits_to_keep(monkeypatch):
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _FakeModel()
    model.supports_logits_to_keep = True
    prompt_prefill = batcher._PromptPrefill(
        model=model,
        uid=1,
        input_ids=[1, 2, 3],
        max_tokens=1,
        top_logprobs=0,
        sampler=_argmax_sampler,
        logits_processors=[],
        inputs_embeds=mx.zeros((1, 3, 2), dtype=mx.float32),
        prompt_kwargs={},
        prefix_cache_save_state=_prefix_cache_save_states(1)[0],
        prefill_step_size=2,
    )

    assert prompt_prefill.prompt_step() == 2
    generation_batch, _ = prompt_prefill.generate(lambda _token: False)
    generation_batch.next()

    assert [call["logits_to_keep"] for call in model.calls] == [1, 1, 1]


def test_generation_batch_applies_per_sequence_processors_and_top_logprobs():
    """Processors are per-row, and sampled token metadata follows decode-ahead."""
    model = _FakeModel()
    history_processor = _HistoryProcessor(token=3)
    second_history_processor = _HistoryProcessor(token=4)
    batch = GenerationBatch(
        model=model,
        uids=[10, 11],
        inputs=mx.array([1, 2], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache()],
        samplers=[_argmax_sampler, _argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3, 3],
        top_logprobs_k=2,
        all_tokens=[[100], [200]],
        logits_processors=[[history_processor], [second_history_processor]],
        prefix_cache_save_states=_prefix_cache_save_states(2),
    )

    first = batch.next()
    second = batch.next()

    assert [response.token for response in first] == [1, 2]
    assert [response.token for response in second] == [3, 4]
    assert [response.top_logprobs[0][0] for response in second] == [3, 4]
    assert history_processor.calls[0] == [100, 1]
    assert second_history_processor.calls[0] == [200, 2]


def test_int_last_token_processor_uses_full_context_call():
    """Structured processors do not receive MLX arrays via process_last_token."""
    processor = _IntLastTokenProcessor(token=5)

    logits = batcher._apply_logits_processors(
        mx.zeros((1, 8), dtype=mx.float32),
        [[100]],
        [[processor]],
        last_tokens=mx.array([2], dtype=mx.int32),
    )

    assert processor.calls == [[100, 2]]
    assert processor.last_token_calls == []
    assert mx.argmax(logits, axis=-1).tolist() == [5]


def test_gemma4_reasoning_guard_uses_mlx_last_token_without_mutating_context():
    processor = Gemma4ReasoningGuardLogitsProcessor(
        reasoning_open=False,
        reasoning_start_token_ids=(1, 2),
        reasoning_end_token_ids=(3,),
        tool_call_start_token_id=5,
        tool_grammar=_NoopToolGrammar(),
        eos_token_ids=(0,),
        whitespace_token_ids=(13,),
    )
    processor(mx.array([1], dtype=mx.int32), mx.zeros((1, 16), dtype=mx.float32))
    token_context = [[100]]

    logits = batcher._apply_logits_processors(
        mx.zeros((1, 16), dtype=mx.float32),
        token_context,
        [[processor]],
        last_tokens=mx.array([2], dtype=mx.int32),
    )

    assert token_context == [[100]]
    assert logits[:, 5].tolist() == [-float("inf")]


def test_generation_batch_finish_returns_cache_tokens_and_rope_delta():
    """A finished row returns the mutable cache state needed by hot restore."""
    prompt_cache = [_FakeBatchCache()]
    batch = GenerationBatch(
        model=_FakeModel(),
        uids=[7],
        inputs=mx.array([9], dtype=mx.int32),
        prompt_cache=prompt_cache,
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[1],
        all_tokens=[[1, 2]],
        rope_deltas=mx.array([5], dtype=mx.int32),
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )

    response = batch.next()[0]

    assert response.finish_reason == "length"
    assert response.all_tokens == [1, 2, 9]
    assert response.prompt_cache[0].name == "cache:0"
    assert response.rope_deltas.tolist() == [[5]]


def test_generation_batch_extracts_vlm_scalar_caches_from_vlm_batches():
    keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    values = keys + 10
    cache_cases = [
        (KVCache(), BatchKVCache),
        (RotatingKVCache(max_size=8, keep=0), BatchRotatingKVCache),
    ]

    for scalar_cache, batch_cache_type in cache_cases:
        scalar_cache.update_and_fetch(keys, values)
        batch_cache = scalar_cache.merge([scalar_cache])
        generation_batch = GenerationBatch(
            model=_FakeModel(),
            uids=[7],
            inputs=mx.array([9], dtype=mx.int32),
            prompt_cache=[batch_cache],
            samplers=[_argmax_sampler],
            stop_criteria=lambda _token: False,
            max_tokens=[1],
            all_tokens=[[1, 2]],
            logits_processors=[[]],
            prefix_cache_save_states=_prefix_cache_save_states(1),
        )

        extracted = generation_batch.extract_cache(0)

        assert type(batch_cache) is batch_cache_type
        assert extracted is not None
        assert type(extracted[0]) is type(scalar_cache)
        assert extracted[0].meta_state == scalar_cache.meta_state
        assert extracted[0].state[0].tolist() == scalar_cache.state[0].tolist()
        assert extracted[0].state[1].tolist() == scalar_cache.state[1].tolist()


def test_generation_batch_extends_mixed_rope_rows_without_broadcasting():
    """Appending text-only work to image work gives each row its own RoPE delta."""
    model = _FakeModel()
    batch = GenerationBatch(
        model=model,
        uids=[1],
        inputs=mx.array([5], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache("image")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[5]],
        rope_deltas=mx.array([9], dtype=mx.int32),
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )
    text_only = GenerationBatch(
        model=model,
        uids=[2],
        inputs=mx.array([6], dtype=mx.int32),
        prompt_cache=[_FakeBatchCache("text")],
        samplers=[_argmax_sampler],
        stop_criteria=lambda _token: False,
        max_tokens=[3],
        all_tokens=[[6]],
        logits_processors=[[]],
        prefix_cache_save_states=_prefix_cache_save_states(1),
    )

    batch.append_prefilled_sequence(text_only)
    batch.next()

    assert model.calls[-1]["rope_deltas"] == [[9], [0]]


def test_capture_rope_deltas_keeps_qwen3_5_text_only_none():
    """Qwen3.5 text-only decode stays on the fast text RoPE path."""
    qwen3_5_model = SimpleNamespace(
        language_model=SimpleNamespace(model_type="qwen3_5_vl", _rope_deltas=None)
    )
    qwen_model = SimpleNamespace(
        language_model=SimpleNamespace(model_type="qwen2_vl", _rope_deltas=None)
    )

    assert batcher._capture_rope_deltas(qwen3_5_model, rows=2) is None
    assert batcher._capture_rope_deltas(qwen_model, rows=2).tolist() == [[0], [0]]


def test_prompt_prefill_prefers_request_owned_rope_deltas(monkeypatch):
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _FakeModel()
    model.language_model = SimpleNamespace(
        model_type="qwen3_5_vl",
        _rope_deltas=mx.array([99], dtype=mx.int32),
    )
    prompt_prefill = batcher._PromptPrefill(
        model=model,
        uid=1,
        input_ids=[1, 2],
        max_tokens=1,
        top_logprobs=0,
        sampler=_argmax_sampler,
        logits_processors=[],
        inputs_embeds=mx.zeros((1, 2, 2), dtype=mx.float32),
        prompt_kwargs={"rope_deltas": mx.array([7], dtype=mx.int32)},
        prefix_cache_save_state=_prefix_cache_save_states(1)[0],
    )

    generation_batch, _ = prompt_prefill.generate(lambda _token: False)

    assert generation_batch._rope_deltas.tolist() == [[7]]


@pytest.mark.parametrize("use_mrope", [False, True], ids=["text", "mrope"])
def test_batch_generator_slices_position_ids_and_saves_prefill_boundaries(
    monkeypatch,
    use_mrope,
):
    """Chunked prefill keeps text and MRoPE positions aligned with embeds."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=256,
    )
    snapshots = []
    prompt = list(range(513))
    if use_mrope:
        position_ids = mx.array(
            [
                [list(range(513))],
                [list(range(1000, 1513))],
                [list(range(2000, 2513))],
            ],
            dtype=mx.int32,
        )
    else:
        position_ids = mx.array([list(range(513))], dtype=mx.int32)

    prefix_chunks = build_prefix_cache_chunks(prompt, [])

    def save_snapshot(cache, chunks, start_chunk_idx, end_chunk_idx, snapshot_len):
        snapshots.append((cache, chunks, start_chunk_idx, end_chunk_idx, snapshot_len))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"position_ids": position_ids},
            prefix_cache_chunks=prefix_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            prompt_cache_save_callback=save_snapshot,
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [256, 256, 1]
    assert [call["position_ids"] for call in model.calls] == [
        position_ids[..., :256].tolist(),
        position_ids[..., 256:512].tolist(),
        position_ids[..., 512:].tolist(),
    ]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len)
        for _, _, start_chunk_idx, end_chunk_idx, snapshot_len in snapshots
    ] == [
        (0, 1, 256),
        (1, 2, 512),
    ]


@pytest.mark.parametrize("token_type_key", ["mm_token_type_ids", "token_type_ids"])
def test_batch_generator_uses_gemma4_token_types_when_cache_span_is_coarse(
    monkeypatch,
    token_type_key,
):
    """Cache fallback spans do not become protected attention-mask spans."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(1_300))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 600:700] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={token_type_key: mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=0, end=len(prompt), image_hash="image")],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [512, 512, 276]
    assert model.calls[0][token_type_key] is None
    assert model.calls[1][token_type_key] == [[0] * 600 + [1] * 100 + [0] * 324]
    assert model.calls[2][token_type_key] is None


def test_batch_generator_gemma4_without_token_types_chunks_normally(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=0, end=len(prompt), image_hash="image")],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4, 4, 2]
    assert all(call["mm_token_type_ids"] is None for call in model.calls)
    assert all(call["token_type_ids"] is None for call in model.calls)


def test_batch_generator_uses_image_safe_non_aligned_prefill(monkeypatch):
    """Gemma can end at an image boundary and still backfill cache chunks."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(1_200))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 400:600] = 1
    prefix_cache_chunks = build_prefix_cache_chunks(
        prompt,
        [PromptImageSpan(start=400, end=600, image_hash="image")],
    )
    snapshot_lengths = []

    def save_snapshot(_cache, _chunks, _start_idx, _end_idx, snapshot_len):
        snapshot_lengths.append(snapshot_len)

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=prefix_cache_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=400, end=600, image_hash="image")],
            prompt_cache_save_callback=save_snapshot,
        )

        for _ in range(4):
            generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [256, 344, 424, 176]
    assert snapshot_lengths == [256, 600, 1024]
    assert model.calls[0]["mm_token_type_ids"] is None
    assert model.calls[1]["mm_token_type_ids"] == [[0] * 400 + [1] * 200]
    assert model.calls[2]["mm_token_type_ids"] is None
    assert model.calls[3]["mm_token_type_ids"] is None


def test_batch_generator_splits_overlapping_gemma4_cache_envelopes(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=256,
    )
    prompt = list(range(900))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 200:300] = 1
    mm_token_type_ids[:, 500:600] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[
                PromptImageSpan(start=200, end=300, image_hash="first"),
                PromptImageSpan(start=500, end=600, image_hash="second"),
            ],
        )

        for _ in range(4):
            generator.next()
    finally:
        generator.close()

    call_lengths = [len(call["input_ids"][0]) for call in model.calls]
    assert call_lengths == [200, 256, 256, 188]
    assert max(call_lengths) <= 256

    boundary = 0
    for call_length in call_lengths[:-1]:
        boundary += call_length
        assert not 200 < boundary < 300
        assert not 500 < boundary < 600


def test_batch_generator_keeps_image_longer_than_prefill_step_whole(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=256,
    )
    prompt = list(range(500))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 100:400] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=100, end=400, image_hash="image")],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [100, 300, 100]


def test_batch_generator_splits_neighboring_gemma4_image_runs(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=256,
    )
    prompt = list(range(700))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 100:150] = 1
    mm_token_type_ids[:, 300:350] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[
                PromptImageSpan(start=100, end=150, image_hash="first"),
                PromptImageSpan(start=300, end=350, image_hash="second"),
            ],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [256, 256, 188]


def test_batch_generator_keeps_multiple_gemma4_images_whole(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(2_000))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 400:600] = 1
    mm_token_type_ids[:, 1_100:1_300] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[
                PromptImageSpan(start=400, end=600, image_hash="first"),
                PromptImageSpan(start=1_100, end=1_300, image_hash="second"),
            ],
        )

        for _ in range(5):
            generator.next()
    finally:
        generator.close()

    call_lengths = [len(call["input_ids"][0]) for call in model.calls]
    assert call_lengths == [256, 512, 256, 512, 464]
    boundary = 0
    for call_length in call_lengths[:-1]:
        boundary += call_length
        assert boundary % 256 == 0
        assert not 400 < boundary < 600
        assert not 1_100 < boundary < 1_300


def test_batch_generator_long_gemma4_prompt_never_uses_visual_prefix(monkeypatch):
    """An image near 39K cannot turn prefill into a square 39K model call."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=2048,
    )
    prompt_len = 40_001
    image_start = 38_000
    image_end = image_start + 1_120
    prompt = list(range(prompt_len))
    mm_token_type_ids = mx.concatenate(
        [
            mx.zeros((1, image_start), dtype=mx.int32),
            mx.ones((1, image_end - image_start), dtype=mx.int32),
            mx.zeros((1, prompt_len - image_end), dtype=mx.int32),
        ],
        axis=1,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, prompt_len, 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[
                PromptImageSpan(
                    start=image_start,
                    end=image_end,
                    image_hash="image",
                )
            ],
        )

        while sum(len(call["input_ids"][0]) for call in model.calls) < prompt_len:
            generator.next()
    finally:
        generator.close()

    call_lengths = [len(call["input_ids"][0]) for call in model.calls]
    assert sum(call_lengths) == prompt_len
    assert max(call_lengths) <= 2048

    boundary = 0
    for call_length in call_lengths[:-1]:
        boundary += call_length
        assert boundary % 256 == 0
        assert not image_start < boundary < image_end

    assert sum(call["mm_token_type_ids"] is not None for call in model.calls) == 1


def test_batch_generator_chunks_gemma4_text_only_normally(monkeypatch):
    """Gemma4 unified text-only prompts keep the configured prefill size."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={
                "mm_token_type_ids": mx.zeros((1, len(prompt)), dtype=mx.int32)
            },
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4, 4, 2]
    assert all(call["mm_token_type_ids"] is None for call in model.calls)


def test_batch_generator_keeps_trailing_gemma4_image_whole(monkeypatch):
    """The image run remains whole with its required trailing EOI token."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(1_000))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 700:900] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=700, end=900, image_hash="image")],
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [512, 488]


def test_batch_generator_pads_gemma4_token_types_after_restore(monkeypatch):
    """A new-image suffix can build masks against restored cached prefix keys."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(600))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 88:188] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=list(range(512)),
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=600, end=700, image_hash="image")],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [512]
    assert model.calls[0]["mm_token_type_ids"] == [[0] * 600 + [1] * 100 + [0] * 324]


def test_batch_generator_restores_immediately_before_gemma4_image(monkeypatch):
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(800))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, :188] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=list(range(512)),
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=512, end=700, image_hash="image")],
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [512, 288]
    assert model.calls[0]["mm_token_type_ids"] == [[0] * 512 + [1] * 188 + [0] * 324]
    assert model.calls[1]["mm_token_type_ids"] is None


def test_batch_generator_pads_gemma4_token_types_for_final_prefill(monkeypatch):
    """Final prefill also needs key-length token types when restored before image."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(3))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mx.array([[0, 1, 1]], dtype=mx.int32)},
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=[100, 101, 102, 103, 104],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=6, end=8, image_hash="image")],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [3]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 0, 1, 1]]


def test_batch_generator_chunks_bidir_gemma4_around_images(monkeypatch):
    """Non-unified Gemma 4 models use image-boundary-aware chunking."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=512,
    )
    prompt = list(range(1_000))
    mm_token_type_ids = mx.zeros((1, len(prompt)), dtype=mx.int32)
    mm_token_type_ids[:, 400:600] = 1

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=400, end=600, image_hash="image")],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [256, 512, 232]


def test_batch_generator_chunks_non_bidir_gemma4_normally(monkeypatch):
    """Gemma4 without bidirectional visual attention keeps normal chunking."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [_FakeBatchCache()],
    )
    model = _gemma4_non_bidir_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(10))
    mm_token_type_ids = mx.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
        dtype=mx.int32,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"mm_token_type_ids": mm_token_type_ids},
            prefix_cache_chunks=[],
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[PromptImageSpan(start=5, end=8, image_hash="image")],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4, 4, 2]


def test_batch_generator_aligns_restored_prefill_only_for_cache_saves(monkeypatch):
    """Restored prefill alignment is only worth paying for disk snapshots."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())

    def call_lengths(prompt_cache_save_callback, steps: int):
        model = _FakeModel()
        generator = BatchGenerator(
            model=model,
            stop_criteria=lambda _token: False,
            prefill_step_size=4,
        )
        prompt = [10, 11, 12, 13, 14, 15, 16]

        try:
            generator.insert(
                prompt,
                inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
                sampler=_argmax_sampler,
                logits_processors=[],
                prompt_kwargs={},
                prefix_cache_chunks=[],
                image_spans=[],
                cache=[_FakeScalarCache()],
                all_tokens=[0, 1],
                next_prefix_cache_chunk_idx=0,
                prompt_cache_save_callback=prompt_cache_save_callback,
            )

            for _ in range(steps):
                generator.next()
        finally:
            generator.close()

        return [len(call["input_ids"][0]) for call in model.calls]

    assert call_lengths(None, steps=2) == [4, 3]
    assert call_lengths(lambda *_args: None, steps=3) == [2, 4, 1]


def test_batch_generator_state_cache_lands_on_reusable_tail_boundary(monkeypatch):
    """Opaque state caches need an exact checkpoint at the final 256 boundary."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "make_prompt_cache",
        lambda _model: [ArraysCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=2048,
    )
    snapshots = []
    prompt = list(range(1795))
    prefix_chunks = build_prefix_cache_chunks(prompt, [])

    def save_snapshot(cache, chunks, start_chunk_idx, end_chunk_idx, snapshot_len):
        snapshots.append((cache, chunks, start_chunk_idx, end_chunk_idx, snapshot_len))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={},
            prefix_cache_chunks=prefix_chunks,
            all_tokens=[],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
            prompt_cache_save_callback=save_snapshot,
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [1792, 3]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len)
        for _, _, start_chunk_idx, end_chunk_idx, snapshot_len in snapshots
    ] == [(0, 7, 1792)]
