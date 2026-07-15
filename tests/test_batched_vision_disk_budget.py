import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache import disk_budget
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


_GIB = 1024 * 1024 * 1024


class _DiskUsage:
    def __init__(self, free: int):
        self.free = free


def _set_free_disk(monkeypatch, free_bytes: int) -> None:
    monkeypatch.setattr(
        disk_budget.shutil,
        "disk_usage",
        lambda path: _DiskUsage(free_bytes),
    )


def _kv_cache(prefix_len: int):
    cache = KVCache()
    keys = mx.ones((1, 1, prefix_len, 1), dtype=mx.float32)
    cache.state = (keys, keys)
    return cache


def _rotating_cache(prefix_len: int, window_size: int):
    cache = RotatingKVCache(max_size=window_size, keep=0)
    keys = mx.ones((1, 1, prefix_len, 1), dtype=mx.float32)
    cache.update_and_fetch(keys, keys)
    return cache


def _arrays_cache():
    cache = ArraysCache(1)
    cache[0] = mx.ones((2, 3), dtype=mx.float32)
    return cache


def test_provisional_budget_uses_free_disk_guard(monkeypatch, tmp_path):
    """Provisional budget disables low-disk stores and otherwise caps at 30 GiB."""
    _set_free_disk(monkeypatch, 9 * _GIB)
    assert disk_budget.provisional_cache_store_budget_bytes(tmp_path) == 0

    _set_free_disk(monkeypatch, 80 * _GIB)
    assert disk_budget.provisional_cache_store_budget_bytes(tmp_path) == 20 * _GIB

    _set_free_disk(monkeypatch, 200 * _GIB)
    assert disk_budget.provisional_cache_store_budget_bytes(tmp_path) == 30 * _GIB


def test_final_budget_requires_max_kv_size(monkeypatch, tmp_path):
    """Without max_kv_size, the cache store keeps its provisional budget."""
    _set_free_disk(monkeypatch, 200 * _GIB)

    assert (
        disk_budget.final_cache_store_budget_bytes(tmp_path, [_kv_cache(4)], None)
        is None
    )


def test_final_budget_scales_cache_kinds(monkeypatch, tmp_path, caplog):
    """KV/SWA scale to max size, while state scales by checkpoint count."""
    _set_free_disk(monkeypatch, 200 * _GIB)
    caplog.set_level("INFO")
    max_kv_size = 16
    prompt_cache = [
        _kv_cache(4),
        _rotating_cache(prefix_len=4, window_size=8),
        _arrays_cache(),
    ]

    budget = disk_budget.final_cache_store_budget_bytes(
        tmp_path,
        prompt_cache,
        max_kv_size,
    )

    kv_bytes = 2 * max_kv_size * 4
    rotating_bytes = 2 * max_kv_size * 4
    arrays_bytes = 1 * 2 * 3 * 4
    assert budget == kv_bytes + rotating_bytes + arrays_bytes
    assert (
        "stage=final target_tokens=16 desired_gib=0.00 cap_gib=0.00 "
        "limiter=max_kv_size" in caplog.text
    )

    budget = disk_budget.final_cache_store_budget_bytes(
        tmp_path,
        [_arrays_cache()],
        2 * DEFAULT_PREFIX_CHUNK_SIZE,
    )

    assert budget == 2 * 2 * 3 * 4


def test_final_budget_respects_free_disk_guard(monkeypatch, tmp_path):
    """Final budget still disables low-disk stores and caps at free disk / 4."""
    prompt_cache = [_kv_cache(4)]

    _set_free_disk(monkeypatch, 9 * _GIB)
    assert disk_budget.final_cache_store_budget_bytes(tmp_path, prompt_cache, 16) == 0

    _set_free_disk(monkeypatch, 12 * _GIB)
    assert (
        disk_budget.final_cache_store_budget_bytes(tmp_path, prompt_cache, 2**31)
        == 3 * _GIB
    )
