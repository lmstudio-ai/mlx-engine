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


class _LastTokenProcessor:
    def __init__(self, token: int):
        self.token = token
        self.calls = []

    def process_last_token(self, last_token, logits):
        self.calls.append(last_token)
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
        self.merge_calls = []

    def merge(self, caches):
        self.merge_calls.append(caches)
        return _FakeBatchCache(f"merged:{self.name}")


class _FakeModel:
    def __init__(self):
        self.calls = []

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
            }
        )
        batch_size, seq_len = input_ids.shape
        return SimpleNamespace(logits=_logits(batch_size, seq_len))


def test_generation_batch_applies_per_sequence_processors_and_top_logprobs():
    """Processors are per-row, and sampled token metadata follows decode-ahead."""
    model = _FakeModel()
    history_processor = _HistoryProcessor(token=3)
    last_token_processor = _LastTokenProcessor(token=4)
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
        logits_processors=[[history_processor], [last_token_processor]],
        prefix_cache_save_states=_prefix_cache_save_states(2),
    )

    first = batch.next()
    second = batch.next()

    assert [response.token for response in first] == [1, 2]
    assert [response.token for response in second] == [3, 4]
    assert [response.top_logprobs[0][0] for response in second] == [3, 4]
    assert history_processor.calls[0] == [100, 1]
    assert last_token_processor.calls[0] == 2


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


def test_batch_generator_slices_position_ids_and_saves_prefill_boundaries(
    monkeypatch,
):
    """Chunked prefill keeps Qwen MRoPE positions aligned with sliced embeds."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "_make_cache",
        lambda _model, _padding: [_FakeBatchCache()],
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


def test_batch_generator_aligns_restored_prefill_to_step_boundary(monkeypatch):
    """A restored prefix may take one short step before large prefill resumes."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "_make_cache",
        lambda _model, _padding: [_FakeBatchCache()],
    )
    model = _FakeModel()
    generator = BatchGenerator(
        model=model,
        stop_criteria=lambda _token: False,
        prefill_step_size=4,
    )
    prompt = [10, 11, 12, 13, 14, 15, 16]
    restored_tokens = [0, 1]

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
            all_tokens=restored_tokens,
            next_prefix_cache_chunk_idx=0,
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [len(call["input_ids"][0]) for call in model.calls] == [2, 4, 1]


def test_batch_generator_state_cache_lands_on_reusable_tail_boundary(monkeypatch):
    """Opaque state caches need an exact checkpoint at the final 256 boundary."""
    monkeypatch.setattr(batcher, "wired_limit", lambda _model: contextlib.nullcontext())
    monkeypatch.setattr(
        batcher,
        "_make_cache",
        lambda _model, _padding: [ArraysCache()],
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
