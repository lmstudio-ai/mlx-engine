import pytest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_engine.model_kit.batched_vision.prompt_cache.blob_store import (
    TemporarySafetensorBlobStore,
)
from mlx_lm.models.cache import ArraysCache, RotatingKVCache


@pytest.fixture
def blob_store(tmp_path):
    store = TemporarySafetensorBlobStore(tmp_path)
    yield store
    store.close()


def _snapshot(caches):
    # Match the production record format: cache arrays plus cache metadata.
    arrays = dict(tree_flatten([cache.state for cache in caches]))
    metadata = dict(
        tree_flatten(
            [
                [cache.meta_state for cache in caches],
                [type(cache).__name__ for cache in caches],
            ]
        )
    )
    return arrays, metadata


def _rotating_cache():
    cache = RotatingKVCache(max_size=8, keep=0)
    keys = mx.array([[[[1.0], [2.0], [3.0], [4.0]]]])
    values = keys + 10
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    return cache


def _arrays_cache():
    cache = ArraysCache(2)
    cache[0] = mx.array([[1.0, 2.0]])
    cache[1] = mx.array([[3.0, 4.0]])
    mx.eval(cache.state)
    return cache


def _cache_state_values(cache):
    return [(key, array.tolist()) for key, array in tree_flatten(cache.state)]


@pytest.mark.parametrize(
    "cache_factory",
    [_rotating_cache, _arrays_cache],
    ids=["rotating", "arrays"],
)
def test_blob_store_round_trips_cache_record(blob_store, cache_factory):
    cache = cache_factory()

    # Store and load through the public blob-store API only.
    size = blob_store.put("record", *_snapshot([cache]))
    loaded = blob_store.load_record("record")

    assert blob_store.exists("record")
    assert blob_store.size("record") == size
    assert type(loaded[0]) is type(cache)
    assert loaded[0].meta_state == cache.meta_state
    assert _cache_state_values(loaded[0]) == _cache_state_values(cache)


def test_blob_store_delete_removes_record_and_store_remains_usable(blob_store):
    blob_store.put("old", *_snapshot([_arrays_cache()]))

    # Deleting an evicted record must not poison later writes.
    blob_store.delete("old")
    blob_store.put("new", *_snapshot([_rotating_cache()]))

    assert not blob_store.exists("old")
    assert type(blob_store.load_record("new")[0]) is RotatingKVCache
