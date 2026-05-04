import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    PromptCacheRestorePlan,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    VlmPromptCacheCoordinator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    LoadedPromptState,
)
from mlx_lm.models.cache import KVCache


class _FakeCacheStore:
    def __init__(self, *, disk_prefix_len: int | None = None):
        self.restore_plan = (
            None
            if disk_prefix_len is None
            else PromptCacheRestorePlan(
                cached_prefix_len=disk_prefix_len,
                chunks=[],
                record_keys_by_chunk_key={},
            )
        )
        self.loaded_state = (
            None
            if disk_prefix_len is None
            else LoadedPromptState(
                cached_prefix_len=disk_prefix_len,
                prompt_cache=["disk-cache"],
            )
        )
        self.recorded_tokens = []
        self.loaded_plans = []

    def plan_longest_prefix_restore(self, prompt_input_ids, image_spans):
        return self.restore_plan

    def load_restore_plan(self, plan):
        self.loaded_plans.append(plan)
        return self.loaded_state

    def record_restore_tokens(self, *, hit_tokens: int, miss_tokens: int) -> None:
        self.recorded_tokens.append((hit_tokens, miss_tokens))

    def can_store_records(self) -> bool:
        return True


def _coordinator(cache_store):
    enqueued_saves = []
    coordinator = VlmPromptCacheCoordinator(cache_store, enqueued_saves.append)
    return coordinator, enqueued_saves


def _kv_cache(prefix_len: int):
    cache = KVCache()
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    cache.state = (keys, keys + 1000)
    return cache


def test_coordinator_restores_exact_hot_prefix():
    """A completed hot cache serves an immediate follow-up and preserves rope."""
    cache_store = _FakeCacheStore()
    coordinator, _ = _coordinator(cache_store)
    prompt_cache = [_kv_cache(512)]
    rope_deltas = object()

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(512)),
        image_spans=[],
        prompt_cache=prompt_cache,
        rope_deltas=rope_deltas,
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(512)) + [999],
        image_spans=[],
    )

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache is prompt_cache
    assert restored.rope_deltas is rope_deltas
    assert cache_store.recorded_tokens == [(512, 0)]


def test_coordinator_trims_hot_branch_and_drops_rope():
    """A branch from hot cache trims reusable KV and recomputes rope state."""
    cache_store = _FakeCacheStore()
    coordinator, _ = _coordinator(cache_store)
    prompt_cache = [_kv_cache(768)]

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(768)),
        image_spans=[],
        prompt_cache=prompt_cache,
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(512)) + [-1],
        image_spans=[],
    )
    kv_keys, _ = prompt_cache[0].state
    mx.eval(kv_keys)

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache is prompt_cache
    assert restored.rope_deltas is None
    assert kv_keys.shape[2] == 512
    assert cache_store.recorded_tokens == [(512, 0)]


def test_coordinator_prefers_longer_disk_restore():
    """Disk wins when it can restore more tokens than the hot cache."""
    cache_store = _FakeCacheStore(disk_prefix_len=512)
    coordinator, _ = _coordinator(cache_store)

    coordinator.store_hot_prompt_cache(
        prompt_input_ids=list(range(256)),
        image_spans=[],
        prompt_cache=["hot-cache"],
        rope_deltas=object(),
    )
    restored = coordinator.restore(
        prompt_input_ids=list(range(700)),
        image_spans=[],
    )

    assert restored is not None
    assert restored.cached_prefix_len == 512
    assert restored.prompt_cache == ["disk-cache"]
    assert restored.rope_deltas is None
    assert cache_store.loaded_plans == [cache_store.restore_plan]
    assert cache_store.recorded_tokens == [(512, 187)]
