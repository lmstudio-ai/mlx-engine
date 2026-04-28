from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx.utils import tree_flatten


RECORD_KIND_KV_DELTA = "kv_delta"
RECORD_KIND_ROTATING_DELTA = "rotating_delta"
RECORD_KIND_STATE_CHECKPOINT = "state_checkpoint"
RECORD_WRITE_ORDER = (
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
)
PAYLOAD_KIND_BOUNDARY = RECORD_KIND_STATE_CHECKPOINT
PAYLOAD_KIND_KV_DELTA = RECORD_KIND_KV_DELTA


def serialize_rope_deltas(rope_deltas: Optional[Any]) -> Optional[Any]:
    if rope_deltas is None:
        return None

    # Keep the spill payload on disk; tiny RoPE state can stay process-local.
    return mx.array(rope_deltas, dtype=mx.int32).tolist()


def deserialize_rope_deltas(serialized_rope_deltas: Optional[Any]) -> Optional[Any]:
    if serialized_rope_deltas is None:
        return None

    return mx.array(serialized_rope_deltas, dtype=mx.int32)


def is_kv_cache(cache: Any) -> bool:
    # mlx-vlm re-exports mlx-lm cache classes; keep this name-based so local
    # forks do not need identical module identities.
    return type(cache).__name__ == "KVCache"


def is_rotating_kv_cache(cache: Any) -> bool:
    return type(cache).__name__ == "RotatingKVCache" and getattr(cache, "keep", 0) == 0


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
) -> tuple[list[Any], list[str]]:
    payload_cache = []
    payload_kinds = []
    for cache in prompt_cache:
        if is_kv_cache(cache):
            payload_cache.append(slice_kv_cache(cache, chunk_start, chunk_end))
            payload_kinds.append(PAYLOAD_KIND_KV_DELTA)
        elif is_rotating_kv_cache(cache):
            payload_cache.append(slice_rotating_kv_cache(cache, chunk_start, chunk_end))
            payload_kinds.append(RECORD_KIND_ROTATING_DELTA)
        else:
            # Opaque array-state caches stay as exact boundary checkpoints for V1.
            payload_cache.append(cache)
            payload_kinds.append(PAYLOAD_KIND_BOUNDARY)

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
    if len(chunk_prompt_caches) == 1:
        return chunk_prompt_caches[0]

    assembled = []
    layer_count = len(chunk_prompt_caches[-1])
    for layer_idx in range(layer_count):
        layer_kinds = [metadata.payload_kinds[layer_idx] for metadata in chunk_metadata]
        layer_chunks = [prompt_cache[layer_idx] for prompt_cache in chunk_prompt_caches]
        if all(kind == PAYLOAD_KIND_KV_DELTA for kind in layer_kinds):
            assembled.append(concat_kv_delta_caches(layer_chunks))
        elif all(kind == RECORD_KIND_ROTATING_DELTA for kind in layer_kinds):
            assembled.append(
                concat_rotating_delta_caches(
                    [cache for cache in layer_chunks if cache is not None],
                    chunk_metadata[-1].chunk_end,
                )
            )
        else:
            assembled.append(
                next(cache for cache in reversed(layer_chunks) if cache is not None)
            )

    return assembled


def materialize_prompt_state(
    prompt_cache: list[Any],
    rope_deltas: Optional[Any],
) -> None:
    eval_targets = tree_flatten([cache.state for cache in prompt_cache])
    if rope_deltas is not None:
        eval_targets.append(("rope_deltas", rope_deltas))
    if eval_targets:
        mx.eval([value for _, value in eval_targets])
