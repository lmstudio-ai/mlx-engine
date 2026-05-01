import contextlib
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.batched_vision import batch_generator as batcher
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator,
    GenerationBatch,
)


def _argmax_sampler(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


def _logits(batch_size: int, seq_len: int, vocab_size: int = 8):
    return mx.zeros((batch_size, seq_len, vocab_size), dtype=mx.float32)


def _bump(logits, token: int):
    bump = [0.0] * logits.shape[-1]
    bump[token] = 100.0
    return logits + mx.array([bump], dtype=mx.float32)


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
        prefill_step_size=2,
    )
    snapshots = []
    prompt = [10, 11, 12, 13, 14]
    position_ids = mx.array(
        [
            [[0, 1, 2, 3, 4]],
            [[10, 11, 12, 13, 14]],
            [[20, 21, 22, 23, 24]],
        ],
        dtype=mx.int32,
    )

    try:
        generator.insert(
            prompt,
            inputs_embeds=mx.zeros((1, len(prompt), 2), dtype=mx.float32),
            sampler=_argmax_sampler,
            logits_processors=[],
            prompt_kwargs={"position_ids": position_ids},
            cache_save_points=[2, 4],
            prompt_cache_save_callback=lambda cache, tokens: snapshots.append(
                (cache, tokens)
            ),
        )

        generator.next()
        generator.next()
        generator.next()
    finally:
        generator.close()

    assert [call["n_to_process"] for call in model.calls] == [2, 2, None]
    assert model.calls[0]["position_ids"] == [[[0, 1]], [[10, 11]], [[20, 21]]]
    assert model.calls[1]["position_ids"] == [[[2, 3]], [[12, 13]], [[22, 23]]]
    assert model.calls[2]["position_ids"] == [[[4]], [[14]], [[24]]]
    assert [tokens for _, tokens in snapshots] == [prompt[:2], prompt[:4]]
