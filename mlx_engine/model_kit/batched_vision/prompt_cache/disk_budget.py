"""Disk-budget policy for the VLM prompt cache store.

The cache store starts with a conservative provisional budget, then replaces it
once with a final empirical budget derived from the first real completed prompt
cache. This keeps model-architecture guessing out of the main cache store
implementation.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    record_kind_for_prompt_cache,
)
from mlx.utils import tree_flatten


logger = logging.getLogger(__name__)

_GIB_BYTES = 1024 * 1024 * 1024
_MIN_FREE_BYTES = 10 * _GIB_BYTES
_PROVISIONAL_BUDGET_BYTES = 30 * _GIB_BYTES


def provisional_cache_store_budget_bytes(cache_dir: Path) -> int:
    """Return the provisional cache store budget before a real cache is observed.

    Provisional policy:
    - if free disk is under 10 GiB, disable disk records
    - otherwise allow min(30 GiB, free disk / 4)
    """
    budget_bytes, _ = _apply_disk_space_limit(cache_dir, _PROVISIONAL_BUDGET_BYTES)
    logger.info(
        "VLM prompt cache disk budget: stage=provisional cap_gib=%.2f",
        _bytes_to_gib(budget_bytes),
    )
    return budget_bytes


def final_cache_store_budget_bytes(
    cache_dir: Path,
    prompt_cache: list[Any],
    max_kv_size: int | None,
) -> int | None:
    """Return the final empirical budget from a completed real prompt cache.

    KV and Rotating/SWA records scale to max_kv_size so single-sequence branch
    restores can keep old sliding-window records on disk. Opaque state caches
    scale by cache chunk count because they are exact-boundary checkpoints. The
    provisional 30 GiB ceiling is removed after observation; the final budget is
    still limited to one quarter of currently free disk.
    """
    if max_kv_size is None:
        return None

    desired_bytes = _estimate_model_sized_prompt_cache_bytes(
        prompt_cache,
        max_kv_size,
    )
    budget_bytes, limited_by_free_disk = _apply_disk_space_limit(
        cache_dir,
        desired_bytes,
    )
    logger.info(
        "VLM prompt cache disk budget: stage=final cap_gib=%.2f limiter=%s",
        _bytes_to_gib(budget_bytes),
        "free_disk_space" if limited_by_free_disk else "max_kv_size",
    )
    return budget_bytes


def _apply_disk_space_limit(cache_dir: Path, desired_bytes: int) -> tuple[int, bool]:
    """Apply the global free-disk guard to a desired cache size."""
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < _MIN_FREE_BYTES:
        return 0, True
    desired_bytes = max(0, desired_bytes)
    free_disk_budget = free_bytes // 4
    if free_disk_budget < desired_bytes:
        return free_disk_budget, True
    return desired_bytes, False


def _bytes_to_gib(value: int) -> float:
    return value / _GIB_BYTES


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
        return _scaled_kv_bytes(cache, max_kv_size)

    # Opaque caches are persisted as exact-boundary state checkpoints.
    checkpoint_count = max(1, max_kv_size // DEFAULT_PREFIX_CHUNK_SIZE)
    return checkpoint_count * sum(
        _array_nbytes(array) for _, array in tree_flatten(cache.state)
    )


def _scaled_kv_bytes(cache: Any, target_token_count: int) -> int:
    keys, values = cache.state
    observed_token_count = int(keys.shape[2])
    if observed_token_count <= 0:
        return 0

    observed_bytes = _array_nbytes(keys) + _array_nbytes(values)
    # The final budget should never be smaller than the cache shape we observed.
    target_token_count = max(target_token_count, observed_token_count)
    return (
        observed_bytes * target_token_count + observed_token_count - 1
    ) // observed_token_count


def _array_nbytes(value: Any) -> int:
    return int(getattr(value, "nbytes", 0) or 0)
