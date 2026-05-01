import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.image_spans import (
    image_safe_common_prefix_len,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PendingPromptCacheSave,
    PromptImageSpan,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    PromptCacheRestorePlan,
    VlmPromptCacheStore,
)
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

logger = logging.getLogger(__name__)


@dataclass
class RestoredPromptCache:
    """Prompt cache restored from hot memory or disk for request insertion."""

    cached_prefix_len: int
    prompt_cache: list[Any]
    rope_deltas: Any | None


@dataclass
class _HotPromptCacheEntry:
    """One completed mutable prompt cache kept in memory for fast follow-up."""

    prompt_input_ids: list[int]
    image_spans: list[PromptImageSpan]
    prompt_cache: list[Any]
    rope_deltas: Any | None


@dataclass
class _HotRestorePlan:
    """Taken hot cache plus the prefix length it can serve for this request."""

    entry: _HotPromptCacheEntry
    cached_prefix_len: int


class VlmPromptCacheCoordinator:
    """Coordinates hot-cache reuse and cache-I/O-thread disk cache writes.

    The generation thread records the most recently completed runtime cache.
    Request preparation consumes that hot cache, compares it with disk, then
    restores whichever source can serve the longest prefix.
    """

    def __init__(
        self,
        cache_store: VlmPromptCacheStore,
        # Called after the generation thread prepares immutable cache records.
        # The owner decides how to persist it, usually via cache I/O thread.
        enqueue_pending_save: Callable[[PendingPromptCacheSave], None],
    ):
        self._cache_store = cache_store
        self._enqueue_pending_save = enqueue_pending_save
        self._hot_entry_lock = Lock()
        self._hot_entry: _HotPromptCacheEntry | None = None

    def restore(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> RestoredPromptCache | None:
        """Restore a prefix and record combined hot/disk cache token efficiency."""
        restored = self._restore_best_prefix(
            prompt_input_ids=prompt_input_ids,
            image_spans=image_spans,
        )
        hit_tokens = 0 if restored is None else restored.cached_prefix_len
        prefill_tokens = max(0, len(prompt_input_ids) - 1)
        self._cache_store.record_restore_tokens(
            hit_tokens=hit_tokens,
            miss_tokens=prefill_tokens - hit_tokens,
        )
        return restored

    def _restore_best_prefix(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> RestoredPromptCache | None:
        """Restore the longest reusable prefix from hot memory or disk.

        This runs on the cache I/O thread. We take the one hot cache, plan
        disk, then load whichever source serves more tokens. Ties prefer hot;
        disk plans are loaded immediately so eviction cannot invalidate them.
        """
        try:
            hot_plan = self._plan_hot_restore(
                entry=self._take_hot_entry(),
                prompt_input_ids=prompt_input_ids,
                image_spans=image_spans,
            )
        except Exception:
            # Hot restore is an optimization; generation can recompute on miss.
            logger.debug(
                "Hot prompt cache restore failed; treating it as a cache miss.",
                exc_info=True,
            )
            hot_plan = None

        disk_plan = self._plan_disk_restore(prompt_input_ids, image_spans)
        if hot_plan is None:
            return (
                None if disk_plan is None else self._load_disk_restore_plan(disk_plan)
            )

        if (
            disk_plan is not None
            and disk_plan.cached_prefix_len > hot_plan.cached_prefix_len
        ):
            # Release hot memory before materializing a better disk prefix.
            hot_plan = None
            return self._load_disk_restore_plan(disk_plan)

        try:
            hot_restored = self._load_hot_restore_plan(hot_plan)
        except Exception:
            logger.debug(
                "Hot prompt cache restore failed; treating it as a cache miss.",
                exc_info=True,
            )
            hot_restored = None
        if hot_restored is not None:
            return hot_restored

        if disk_plan is not None:
            return self._load_disk_restore_plan(disk_plan)

        return None

    def _plan_disk_restore(
        self,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> PromptCacheRestorePlan | None:
        try:
            return self._cache_store.plan_longest_prefix_restore(
                prompt_input_ids,
                image_spans,
            )
        except Exception:
            # Cache-store planning is an optimization; generation can recompute.
            logger.debug(
                "Prompt cache store planning failed; treating it as a cache miss.",
                exc_info=True,
            )
            return None

    def _take_hot_entry(self) -> _HotPromptCacheEntry | None:
        with self._hot_entry_lock:
            entry = self._hot_entry
            # A new request makes the old hot cache cold. Take ownership so we
            # never keep an idle cache around while this request recomputes.
            self._hot_entry = None
            return entry

    def _plan_hot_restore(
        self,
        *,
        entry: _HotPromptCacheEntry | None,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> _HotRestorePlan | None:
        if entry is None:
            return None
        target_prefix_len = image_safe_common_prefix_len(
            prompt_input_ids,
            image_spans,
            entry.prompt_input_ids,
            entry.image_spans,
            max_prefix_len=len(prompt_input_ids) - 1,
        )
        if target_prefix_len <= 0:
            return None

        return _HotRestorePlan(
            entry=entry,
            cached_prefix_len=target_prefix_len,
        )

    def _load_hot_restore_plan(
        self,
        plan: _HotRestorePlan,
    ) -> RestoredPromptCache | None:
        entry = plan.entry
        trim_count = len(entry.prompt_input_ids) - plan.cached_prefix_len
        if trim_count > 0:
            if not can_trim_prompt_cache(entry.prompt_cache):
                return None
            trimmed = trim_prompt_cache(entry.prompt_cache, trim_count)
            if trimmed != trim_count:
                return None

        return RestoredPromptCache(
            cached_prefix_len=plan.cached_prefix_len,
            prompt_cache=entry.prompt_cache,
            rope_deltas=None if trim_count > 0 else entry.rope_deltas,
        )

    def _load_disk_restore_plan(
        self,
        disk_plan: PromptCacheRestorePlan,
    ) -> RestoredPromptCache | None:
        try:
            cached_state = self._cache_store.load_restore_plan(disk_plan)
        except Exception:
            logger.debug(
                "Prompt cache store restore failed; treating it as a cache miss.",
                exc_info=True,
            )
            return None

        return RestoredPromptCache(
            cached_prefix_len=cached_state.cached_prefix_len,
            prompt_cache=cached_state.prompt_cache,
            rope_deltas=None,
        )

    def save_points_after(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
        prompt_progress: int,
    ) -> list[int]:
        """Return prompt save points plus one future decode save point.

        prompt_progress is the prefix length already present in the runtime
        cache, usually from a restore, so earlier save points are skipped.
        """
        save_points = []
        last_save_point = 0
        for chunk in build_prefix_cache_chunks(prompt_input_ids, image_spans):
            if chunk.end <= prompt_progress:
                last_save_point = chunk.end
            else:
                save_points.append(chunk.end)

        if save_points:
            last_save_point = save_points[-1]

        if prompt_progress > last_save_point:
            skipped_chunks = (
                prompt_progress - last_save_point
            ) // DEFAULT_PREFIX_CHUNK_SIZE
            last_save_point += skipped_chunks * DEFAULT_PREFIX_CHUNK_SIZE

        save_points.append(last_save_point + DEFAULT_PREFIX_CHUNK_SIZE)
        return save_points

    def make_save_callback(
        self,
        *,
        image_spans: list[PromptImageSpan],
    ) -> Callable[[list[Any], list[int]], None]:
        image_spans = list(image_spans)

        def _callback(
            prompt_cache: list[Any],
            snapshot_input_ids: list[int],
        ) -> None:
            if not self._cache_store.can_store_records():
                return
            snapshot_input_ids = list(snapshot_input_ids)
            chunks = build_prefix_cache_chunks(
                snapshot_input_ids,
                image_spans,
            )
            if not chunks:
                return

            pending_save = self._cache_store.prepare_save(
                chunk=chunks[-1],
                prefix_chunks=chunks,
                prompt_cache=prompt_cache,
            )
            self._enqueue_pending_save(pending_save)

        return _callback

    def store_hot_prompt_cache(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
        prompt_cache: list[Any],
        rope_deltas: Any | None,
    ) -> None:
        """Keep exactly one completed cache hot for the next likely follow-up."""
        with self._hot_entry_lock:
            self._hot_entry = _HotPromptCacheEntry(
                prompt_input_ids=list(prompt_input_ids),
                image_spans=list(image_spans),
                prompt_cache=prompt_cache,
                rope_deltas=rope_deltas,
            )

    def clear_hot_prompt_cache(self) -> None:
        """Drop the idle runtime cache held for follow-up reuse."""
        with self._hot_entry_lock:
            self._hot_entry = None
