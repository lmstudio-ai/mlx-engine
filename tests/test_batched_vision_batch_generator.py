import contextlib
from types import SimpleNamespace

import mlx.core as mx

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


def test_batch_generator_slices_position_ids_and_saves_prefill_boundaries(
    monkeypatch,
):
    """Chunked prefill keeps Qwen MRoPE positions aligned with sliced embeds."""
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
    position_ids = mx.array(
        [
            [list(range(513))],
            [list(range(1000, 1513))],
            [list(range(2000, 2513))],
        ],
        dtype=mx.int32,
    )

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
    assert [len(call["position_ids"][0][0]) for call in model.calls] == [256, 256, 1]
    assert model.calls[0]["position_ids"][0][0][0] == 0
    assert model.calls[0]["position_ids"][0][0][-1] == 255
    assert model.calls[1]["position_ids"][0][0][0] == 256
    assert model.calls[1]["position_ids"][0][0][-1] == 511
    assert model.calls[2]["position_ids"][0][0] == [512]
    assert [
        (start_chunk_idx, end_chunk_idx, snapshot_len)
        for _, _, start_chunk_idx, end_chunk_idx, snapshot_len in snapshots
    ] == [
        (0, 1, 256),
        (1, 2, 512),
    ]


def test_batch_generator_keeps_gemma4_visual_prefix_together(monkeypatch):
    """Gemma4 visual masks need prompt start through last visual token in one call."""
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
    prompt = list(range(14))
    mm_token_type_ids = mx.array(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
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
            image_spans=[],
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8, 4, 2]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 1, 1, 1]]
    assert model.calls[1]["mm_token_type_ids"] == [[0, 0, 0, 0]]


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


def test_batch_generator_does_not_split_gemma4_visual_prompt_tail(monkeypatch):
    """If the last visual token is also the last prompt token, use final prefill."""
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
    prompt = list(range(8))
    mm_token_type_ids = mx.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=mx.int32)

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
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8]


def test_batch_generator_uses_image_spans_without_gemma4_token_types(monkeypatch):
    """Image spans provide a fallback boundary if Gemma4 token types are absent."""
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
            image_spans=[PromptImageSpan(start=5, end=8, image_hash="image")],
        )

        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [8, 2]


def test_batch_generator_pads_gemma4_token_types_after_restore(monkeypatch):
    """A new-image suffix can build masks against restored cached prefix keys."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    model = _gemma4_unified_model()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = list(range(8))

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={
                "mm_token_type_ids": mx.array(
                    [[0, 0, 1, 1, 0, 0, 0, 0]],
                    dtype=mx.int32,
                )
            },
            prefix_cache_chunks=[],
            cache=[_FakeScalarCache()],
            all_tokens=[100, 101, 102, 103, 104],
            next_prefix_cache_chunk_idx=0,
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [4]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 0, 0, 1, 1]]


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
            image_spans=[],
        )

        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [3]
    assert model.calls[0]["mm_token_type_ids"] == [[0, 0, 0, 0, 0, 0, 1, 1]]


def test_batch_generator_does_not_apply_unified_visual_policy_to_gemma4(monkeypatch):
    """Non-unified Gemma4 models keep their existing chunking behavior."""
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
            image_spans=[],
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
