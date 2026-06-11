import pytest

import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptImageSpan,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    make_record_key,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


@pytest.fixture
def cache_store():
    store = VlmPromptCacheStore()
    yield store
    store.close()


def _kv_cache(prefix_len: int):
    cache = KVCache()
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    cache.state = (keys, keys + 1000)
    return cache


def _arrays_cache(prefix_len: int):
    cache = ArraysCache(size=1)
    cache[0] = mx.array([[prefix_len]], dtype=mx.int32)
    return cache


def _rotating_cache(prefix_len: int):
    window_size = 512
    window_start = max(0, prefix_len - window_size)
    keys = mx.arange(window_start, prefix_len, dtype=mx.float32).reshape(
        1,
        1,
        prefix_len - window_start,
        1,
    )
    cache = RotatingKVCache(max_size=window_size, keep=0)
    cache.state = (keys, keys + 2000)
    cache.offset = prefix_len
    cache._idx = keys.shape[2]
    return cache


def _prompt_cache(prefix_len: int):
    return [_kv_cache(prefix_len), _arrays_cache(prefix_len)]


def _rotating_prompt_cache(prefix_len: int):
    return [_kv_cache(prefix_len), _rotating_cache(prefix_len)]


def _save_chunk(
    cache_store,
    chunk,
    chunks,
    prompt_cache,
    *,
    save_state_checkpoint=True,
):
    chunk_idx = chunks.index(chunk)
    # Production does prepare on generation thread, then commit on cache I/O.
    cache_store.commit_pending_save(
        cache_store.prepare_save(
            chunk=chunk,
            prefix_chunks=chunks[: chunk_idx + 1],
            prompt_cache=prompt_cache,
            save_state_checkpoint=save_state_checkpoint,
        )
    )


def _assert_two_chunk_restore(loaded):
    kv_keys, kv_values = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, kv_values, boundary_state)

    assert loaded.cached_prefix_len == 512
    assert kv_keys.shape[2] == 512
    assert kv_keys[0, 0, 0, 0].item() == 0
    assert kv_keys[0, 0, -1, 0].item() == 511
    assert kv_values[0, 0, -1, 0].item() == 1511
    assert boundary_state.item() == 512


def _assert_rotating_restore(loaded):
    kv_keys, _ = loaded.prompt_cache[0].state
    rotating_keys, _ = loaded.prompt_cache[1].state
    mx.eval(kv_keys, rotating_keys)

    assert loaded.cached_prefix_len == 768
    assert kv_keys.shape[2] == 768
    assert rotating_keys.shape[2] == 512
    assert rotating_keys[0, 0, 0, 0].item() == 256
    assert rotating_keys[0, 0, -1, 0].item() == 767


def test_cache_store_commits_and_restores_prefix_records(cache_store):
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # Two saved 256-token chunks should restore a 512-token prefix.
    for chunk in chunks[:2]:
        _save_chunk(cache_store, chunk, chunks, _prompt_cache(chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)

    old_state_record_key = make_record_key(
        chunks[0].key,
        RECORD_KIND_STATE_CHECKPOINT,
    )
    stats = cache_store.snapshot_stats()
    assert old_state_record_key in stats.record_sizes_by_key

    # The old state checkpoint is stale; evicting one record should not hurt.
    cache_store.commit_budget_update(stats.total_bytes - 1)
    stats = cache_store.snapshot_stats()
    assert old_state_record_key not in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)


def test_cache_store_eviction_preserves_shorter_prefix_restore(cache_store):
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    first_chunk, second_chunk = chunks[:2]

    # Budget enough for the first chunk so adding the second evicts the suffix.
    _save_chunk(cache_store, first_chunk, chunks, [_kv_cache(first_chunk.end)])
    first_size = cache_store.snapshot_stats().chunk_sizes_by_key[first_chunk.key]
    cache_store.commit_budget_update(first_size + 64)

    _save_chunk(cache_store, second_chunk, chunks, [_kv_cache(second_chunk.end)])

    stats = cache_store.snapshot_stats()
    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)

    assert stats.chunk_records_available_by_key[first_chunk.key]
    assert not stats.chunk_records_available_by_key.get(second_chunk.key, False)
    assert loaded.cached_prefix_len == first_chunk.end
    assert stats.total_bytes <= stats.max_bytes


def test_cache_store_eviction_preserves_shorter_state_checkpoint_restore(cache_store):
    """Over-budget suffix saves should not destroy the shorter stateful restore."""
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    first_chunk, second_chunk = chunks[:2]

    # This budget can keep the first chunk's KV plus exact opaque checkpoint.
    _save_chunk(cache_store, first_chunk, chunks, _prompt_cache(first_chunk.end))
    first_size = cache_store.snapshot_stats().total_bytes
    cache_store.commit_budget_update(first_size + 64)

    _save_chunk(cache_store, second_chunk, chunks, _prompt_cache(second_chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)

    kv_keys, _ = loaded.prompt_cache[0].state
    boundary_state = loaded.prompt_cache[1][0]
    mx.eval(kv_keys, boundary_state)
    assert loaded.cached_prefix_len == first_chunk.end
    assert kv_keys.shape[2] == first_chunk.end
    assert boundary_state.item() == first_chunk.end


def test_cache_store_skips_state_for_backfilled_chunks(cache_store):
    """Backfilled KV chunks do not advertise stale opaque state checkpoints."""
    prompt_input_ids = list(range(700))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # A 512-token model call can backfill the 256 chunk KV, but its opaque state
    # is exact only at 512.
    _save_chunk(
        cache_store,
        chunks[0],
        chunks,
        _prompt_cache(512),
        save_state_checkpoint=False,
    )
    _save_chunk(cache_store, chunks[1], chunks, _prompt_cache(512))

    stats = cache_store.snapshot_stats()
    old_state_record_key = make_record_key(
        chunks[0].key,
        RECORD_KIND_STATE_CHECKPOINT,
    )
    target_state_record_key = make_record_key(
        chunks[1].key,
        RECORD_KIND_STATE_CHECKPOINT,
    )
    assert old_state_record_key not in stats.record_sizes_by_key
    assert target_state_record_key in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_two_chunk_restore(loaded)


def test_cache_store_does_not_restore_to_prefix_inside_image_span(cache_store):
    """Disk chunks inside images are internal records, not terminal restore points."""
    prompt_input_ids = list(range(900))
    image_spans = [PromptImageSpan(start=200, end=700, image_hash="image")]
    chunks = build_prefix_cache_chunks(prompt_input_ids, image_spans)

    _save_chunk(cache_store, chunks[0], chunks, _prompt_cache(chunks[0].end))
    _save_chunk(cache_store, chunks[1], chunks, _prompt_cache(chunks[1].end))

    assert (
        cache_store.plan_longest_prefix_restore(prompt_input_ids, image_spans) is None
    )

    _save_chunk(cache_store, chunks[2], chunks, _prompt_cache(chunks[2].end))

    restore_plan = cache_store.plan_longest_prefix_restore(
        prompt_input_ids,
        image_spans,
    )

    assert restore_plan is not None
    assert restore_plan.cached_prefix_len == chunks[2].end


def test_cache_store_rotating_restore_uses_target_window(cache_store):
    prompt_input_ids = list(range(900))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])

    # Full KV needs every chunk; rotating KV only needs the target window.
    for chunk in chunks[:3]:
        _save_chunk(cache_store, chunk, chunks, _rotating_prompt_cache(chunk.end))

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_rotating_restore(loaded)

    old_rotating_record_key = make_record_key(
        chunks[0].key,
        RECORD_KIND_ROTATING_DELTA,
    )
    stats = cache_store.snapshot_stats()
    assert old_rotating_record_key in stats.record_sizes_by_key

    # The first rotating record is outside the target SWA window.
    cache_store.commit_budget_update(stats.total_bytes - 1)
    stats = cache_store.snapshot_stats()
    assert old_rotating_record_key not in stats.record_sizes_by_key

    restore_plan = cache_store.plan_longest_prefix_restore(prompt_input_ids, [])
    assert restore_plan is not None
    loaded = cache_store.load_restore_plan(restore_plan)
    _assert_rotating_restore(loaded)
