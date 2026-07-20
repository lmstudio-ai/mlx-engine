import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    assemble_prompt_cache_chunks,
    make_prompt_cache_layout,
    prepare_prompt_cache_records_for_chunk,
    record_kind_for_prompt_cache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
)
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache


def _kv_cache(start: int, end: int):
    cache = KVCache()
    keys = mx.arange(start, end, dtype=mx.float32).reshape(1, 1, end - start, 1)
    cache.state = (keys, keys + 1000)
    return cache


def _rotating_cache(prefix_len: int, *, window_size: int = 8, keep: int = 0):
    window_start = max(0, prefix_len - window_size)
    keys = mx.arange(window_start, prefix_len, dtype=mx.float32).reshape(
        1,
        1,
        prefix_len - window_start,
        1,
    )
    cache = RotatingKVCache(max_size=window_size, keep=keep)
    cache.state = (keys, keys + 2000)
    cache.offset = prefix_len
    cache._idx = keys.shape[2]
    return cache


def _wrapped_rotating_cache(*, window_size: int = 8, decode_tokens: int = 4):
    cache = RotatingKVCache(max_size=window_size, keep=0)
    prompt = mx.arange(window_size, dtype=mx.float32).reshape(1, 1, window_size, 1)
    cache.update_and_fetch(prompt, prompt + 2000)
    for token in range(window_size, window_size + decode_tokens):
        decoded = mx.array([[[[token]]]], dtype=mx.float32)
        cache.update_and_fetch(decoded, decoded + 2000)
    return cache


def _arrays_cache(value: int):
    cache = ArraysCache(size=1)
    cache[0] = mx.array([[value]], dtype=mx.int32)
    return cache


def _values(cache):
    keys, values = cache.state
    mx.eval(keys, values)
    return keys.flatten().tolist(), values.flatten().tolist()


def test_records_classify_and_layout_cache_layers():
    """Record layout preserves the storage policy for each cache layer."""
    caches = [
        _kv_cache(0, 4),
        _rotating_cache(4, window_size=8),
        _rotating_cache(4, keep=2),
        _arrays_cache(4),
        _kv_cache(0, 4),
    ]
    record_kinds = [record_kind_for_prompt_cache(cache) for cache in caches]

    layout = make_prompt_cache_layout(caches, record_kinds)

    assert record_kinds == [
        RECORD_KIND_KV_DELTA,
        RECORD_KIND_ROTATING_DELTA,
        RECORD_KIND_STATE_CHECKPOINT,
        RECORD_KIND_STATE_CHECKPOINT,
        RECORD_KIND_KV_DELTA,
    ]
    assert layout.layer_indices_by_kind == {
        RECORD_KIND_KV_DELTA: [0, 4],
        RECORD_KIND_ROTATING_DELTA: [1],
        RECORD_KIND_STATE_CHECKPOINT: [2, 3],
    }
    assert layout.rotating_window_size == 8


def test_records_prepare_slices_chunk_records():
    """Preparing records slices KV deltas and keeps opaque state exact."""
    state_cache = _arrays_cache(10)
    record_caches, record_kinds = prepare_prompt_cache_records_for_chunk(
        prompt_cache=[
            _kv_cache(0, 10),
            _rotating_cache(10, window_size=8),
            state_cache,
        ],
        chunk_start=2,
        chunk_end=6,
    )

    assert record_kinds == [
        RECORD_KIND_KV_DELTA,
        RECORD_KIND_ROTATING_DELTA,
        RECORD_KIND_STATE_CHECKPOINT,
    ]
    assert type(record_caches[0]) is KVCache
    assert type(record_caches[1]) is RotatingKVCache
    # Both cache kinds should persist the same logical token chunk [2, 6).
    assert _values(record_caches[0]) == (
        [2.0, 3.0, 4.0, 5.0],
        [1002.0, 1003.0, 1004.0, 1005.0],
    )
    assert _values(record_caches[1]) == (
        [2.0, 3.0, 4.0, 5.0],
        [2002.0, 2003.0, 2004.0, 2005.0],
    )
    # Opaque caches are exact-boundary checkpoints, not sliced deltas.
    assert record_caches[2] is state_cache


def test_records_prepare_slices_wrapped_rotating_cache_in_temporal_order():
    """Scalar decode wraps rotating storage, but records stay chronological."""
    record_caches, _ = prepare_prompt_cache_records_for_chunk(
        prompt_cache=[_wrapped_rotating_cache()],
        chunk_start=8,
        chunk_end=12,
    )

    assert _values(record_caches[0]) == (
        [8.0, 9.0, 10.0, 11.0],
        [2008.0, 2009.0, 2010.0, 2011.0],
    )


def test_records_assemble_prompt_cache_chunks():
    """Assembly concatenates KV, trims rotating KV, and keeps latest state."""
    chunks = [
        PromptPrefixChunk(start=0, end=4, key="chunk-0"),
        PromptPrefixChunk(start=4, end=8, key="chunk-1"),
        PromptPrefixChunk(start=8, end=12, key="chunk-2"),
    ]
    chunk_prompt_caches = [
        [_kv_cache(0, 4), _rotating_cache(4, window_size=8), _arrays_cache(4)],
        [_kv_cache(4, 8), _rotating_cache(8, window_size=8), _arrays_cache(8)],
        [_kv_cache(8, 12), _rotating_cache(12, window_size=8), _arrays_cache(12)],
    ]
    layout = make_prompt_cache_layout(
        chunk_prompt_caches[-1],
        [
            RECORD_KIND_KV_DELTA,
            RECORD_KIND_ROTATING_DELTA,
            RECORD_KIND_STATE_CHECKPOINT,
        ],
    )

    assembled = assemble_prompt_cache_chunks(chunk_prompt_caches, chunks, layout)
    state_value = assembled[2][0]
    mx.eval(state_value)

    assert type(assembled[0]) is KVCache
    assert type(assembled[1]) is RotatingKVCache
    assert type(assembled[2]) is ArraysCache
    # Full-attention KV reconstructs the whole cached prefix.
    assert _values(assembled[0]) == (
        list(map(float, range(12))),
        list(map(float, range(1000, 1012))),
    )
    # Rotating KV reconstructs only the target window [4, 12).
    assert _values(assembled[1]) == (
        list(map(float, range(4, 12))),
        list(map(float, range(2004, 2012))),
    )
    assert assembled[1].offset == 12
    assert assembled[1]._idx == 8
    # Opaque state restore uses the latest exact boundary checkpoint.
    assert state_value.item() == 12
