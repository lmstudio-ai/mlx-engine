"""Disk-budget policy for the VLM prompt spill cache.

The spill cache starts with a conservative provisional cap, then replaces it
once with a final empirical cap derived from the first real completed prompt
cache. This keeps model-architecture guessing out of the main spill-cache
implementation.
"""

import shutil
from pathlib import Path
from typing import Any

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    record_kind_for_prompt_cache,
)
from mlx.utils import tree_flatten


_GIB_BYTES = 1024 * 1024 * 1024
_MIN_FREE_BYTES = 10 * _GIB_BYTES
_PROVISIONAL_CAP_BYTES = 30 * _GIB_BYTES


def provisional_spill_cache_cap_bytes(cache_dir: Path) -> int:
    """Return the provisional spill-cache cap before a real cache is observed.

    Provisional policy:
    - if free disk is under 10 GiB, disable disk records
    - otherwise allow min(30 GiB, free disk / 4)
    """
    return _apply_disk_space_limit(cache_dir, _PROVISIONAL_CAP_BYTES)


def final_spill_cache_cap_bytes(
    cache_dir: Path,
    prompt_cache: list[Any],
    max_kv_size: int | None,
) -> int | None:
    """Return the final empirical cap from a completed real prompt cache.

    Full KV layers scale to max_kv_size. Rotating/SWA layers scale only to
    their window. Opaque state caches use observed bytes only. The provisional
    30 GiB cap does not apply after observation; free disk remains the limit.
    """
    if max_kv_size is None:
        return None

    desired_bytes = _estimate_model_sized_prompt_cache_bytes(
        prompt_cache,
        max_kv_size,
    )
    return _apply_disk_space_limit(cache_dir, desired_bytes)


def _apply_disk_space_limit(cache_dir: Path, desired_bytes: int) -> int:
    """Apply the global free-disk guard to a desired cache size."""
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < _MIN_FREE_BYTES:
        return 0
    return min(max(0, desired_bytes), free_bytes // 4)


def _estimate_model_sized_prompt_cache_bytes(
    prompt_cache: list[Any],
    max_kv_size: int,
) -> int:
    return sum(
        _estimate_layer_cache_bytes(cache, max_kv_size) for cache in prompt_cache
    )


def _estimate_layer_cache_bytes(cache: Any, max_kv_size: int) -> int:
    record_kind = record_kind_for_prompt_cache(cache)
    if record_kind == RECORD_KIND_KV_DELTA:
        return _scaled_kv_bytes(cache, max_kv_size)
    if record_kind == RECORD_KIND_ROTATING_DELTA:
        window_size = int(cache.max_size)
        return _scaled_kv_bytes(cache, min(max_kv_size, window_size))

    # Opaque caches are persisted as exact-boundary state checkpoints.
    return sum(_array_nbytes(array) for _, array in tree_flatten(cache.state))


def _scaled_kv_bytes(cache: Any, target_token_count: int) -> int:
    keys, values = cache.state
    observed_token_count = int(keys.shape[2])
    if observed_token_count <= 0:
        return 0

    observed_bytes = _array_nbytes(keys) + _array_nbytes(values)
    # The final cap should never be smaller than the cache shape we observed.
    target_token_count = max(target_token_count, observed_token_count)
    return (
        observed_bytes * target_token_count + observed_token_count - 1
    ) // observed_token_count


def _array_nbytes(value: Any) -> int:
    return int(getattr(value, "nbytes", 0) or 0)
