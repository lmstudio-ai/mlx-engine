from typing import Any, Optional

import mlx.core as mx
from mlx_engine.model_kit.vlm_prompt_cache_types import (
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RecordKind,
    record_kind_for_prompt_cache,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx.utils import tree_flatten


def slice_kv_cache(cache: Any, chunk_start: int, chunk_end: int) -> KVCache:
    keys, values = cache.state
    chunk_cache = KVCache()
    chunk_cache.state = (
        mx.contiguous(keys[..., chunk_start:chunk_end, :]),
        mx.contiguous(values[..., chunk_start:chunk_end, :]),
    )
    return chunk_cache


def slice_rotating_kv_cache(
    cache: Any,
    chunk_start: int,
    chunk_end: int,
) -> RotatingKVCache:
    keys, values = cache.state
    window_start = cache.offset - keys.shape[2]
    local_start = max(0, chunk_start - window_start)
    local_end = min(keys.shape[2], chunk_end - window_start)

    chunk_cache = RotatingKVCache(max_size=cache.max_size, keep=cache.keep)
    chunk_cache.state = (
        mx.contiguous(keys[..., local_start:local_end, :]),
        mx.contiguous(values[..., local_start:local_end, :]),
    )
    chunk_cache.offset = chunk_end
    chunk_cache._idx = local_end - local_start
    return chunk_cache


def prepare_prompt_cache_payload(
    prompt_cache: list[Any],
    chunk_start: int,
    chunk_end: int,
) -> tuple[list[Any], list[RecordKind]]:
    payload_cache = []
    payload_kinds = []
    for cache in prompt_cache:
        record_kind = record_kind_for_prompt_cache(cache)
        if record_kind == RECORD_KIND_KV_DELTA:
            payload_cache.append(slice_kv_cache(cache, chunk_start, chunk_end))
        elif record_kind == RECORD_KIND_ROTATING_DELTA:
            payload_cache.append(slice_rotating_kv_cache(cache, chunk_start, chunk_end))
        else:
            # Opaque array-state caches stay as exact boundary checkpoints for V1.
            payload_cache.append(cache)
        payload_kinds.append(record_kind)

    return payload_cache, payload_kinds


def concat_kv_delta_caches(caches: list[Any]) -> KVCache:
    keys = mx.concatenate([cache.state[0] for cache in caches], axis=2)
    values = mx.concatenate([cache.state[1] for cache in caches], axis=2)
    cache = KVCache()
    cache.state = (mx.contiguous(keys), mx.contiguous(values))
    return cache


def concat_rotating_delta_caches(
    caches: list[Any],
    final_offset: int,
) -> RotatingKVCache:
    keys = mx.concatenate([cache.state[0] for cache in caches], axis=2)
    values = mx.concatenate([cache.state[1] for cache in caches], axis=2)
    max_size = caches[-1].max_size
    keep = caches[-1].keep
    if keys.shape[2] > max_size:
        keys = keys[..., -max_size:, :]
        values = values[..., -max_size:, :]

    cache = RotatingKVCache(max_size=max_size, keep=keep)
    cache.state = (mx.contiguous(keys), mx.contiguous(values))
    cache.offset = final_offset
    cache._idx = keys.shape[2]
    return cache


def assemble_prompt_cache_chunks(
    chunk_prompt_caches: list[list[Any]],
    chunk_metadata: list[Any],
) -> list[Any]:
    """Rebuild one prompt cache from a loadable prefix chunk sequence."""
    if len(chunk_prompt_caches) == 1:
        return chunk_prompt_caches[0]

    assembled = []
    layer_count = len(chunk_prompt_caches[-1])
    for layer_idx in range(layer_count):
        layer_chunks = [prompt_cache[layer_idx] for prompt_cache in chunk_prompt_caches]
        record_kind = chunk_metadata[-1].payload_kinds[layer_idx]
        if record_kind == RECORD_KIND_KV_DELTA:
            # Full-attention layers keep every chunk in prefix order.
            assembled.append(concat_kv_delta_caches(layer_chunks))
        elif record_kind == RECORD_KIND_ROTATING_DELTA:
            # Planner loads only chunks that overlap the target sliding window.
            assembled.append(
                concat_rotating_delta_caches(
                    [cache for cache in layer_chunks if cache is not None],
                    chunk_metadata[-1].chunk_end,
                )
            )
        elif record_kind == RECORD_KIND_STATE_CHECKPOINT:
            # Opaque state caches are only valid at exact saved boundaries.
            assembled.append(
                next(cache for cache in reversed(layer_chunks) if cache is not None)
            )
        else:
            raise ValueError(f"unsupported prompt cache record kind: {record_kind}")

    return assembled


def materialize_prompt_state(
    prompt_cache: list[Any],
    rope_deltas: Optional[Any],
) -> None:
    eval_targets = [
        value for _, value in tree_flatten([cache.state for cache in prompt_cache])
    ]
    if rope_deltas is not None:
        eval_targets.append(rope_deltas)
    if eval_targets:
        mx.eval(eval_targets)
