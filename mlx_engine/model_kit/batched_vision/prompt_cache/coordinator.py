import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.image_spans import (
    image_safe_common_prefix_len,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PendingPromptCacheSave,
    PromptImageSpan,
    PromptPrefixChunk,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    DiskPromptCacheRestorePlan,
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    PromptCacheRecordCoverageError,
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
class _HotRestoreCandidate:
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
        miss_tokens = prefill_tokens - hit_tokens
        lifetime_hit_tokens, lifetime_miss_tokens = (
            self._cache_store.record_restore_tokens(
                hit_tokens=hit_tokens,
                miss_tokens=miss_tokens,
            )
        )
        lifetime_tokens = lifetime_hit_tokens + lifetime_miss_tokens
        lifetime_efficiency = (
            100.0 * lifetime_hit_tokens / lifetime_tokens if lifetime_tokens else 0.0
        )
        logger.info(
            "Prompt cache restore: cached_tokens=%s uncached_tokens=%s "
            "lifetime_efficiency=%.2f%%",
            hit_tokens,
            miss_tokens,
            lifetime_efficiency,
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

        if self._hot_restore_covers_max_prefix_without_trim(
            hot_plan,
            max_prefix_len=len(prompt_input_ids) - 1,
        ):
            return self._load_hot_restore_plan(hot_plan)

        disk_plan = self._plan_disk_restore(prompt_input_ids, image_spans)
        if hot_plan is None:
            return (
                None if disk_plan is None else self._load_disk_restore_plan(disk_plan)
            )

        if (
            disk_plan is not None
            and disk_plan.cached_prefix_len > hot_plan.cached_prefix_len
        ):
            disk_restored = self._load_disk_restore_plan(disk_plan)
            if disk_restored is not None:
                return disk_restored

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

    def _hot_restore_covers_max_prefix_without_trim(
        self,
        hot_plan: _HotRestoreCandidate | None,
        *,
        max_prefix_len: int,
    ) -> bool:
        return (
            hot_plan is not None
            and hot_plan.cached_prefix_len == max_prefix_len
            and len(hot_plan.entry.prompt_input_ids) == hot_plan.cached_prefix_len
        )

    def _plan_disk_restore(
        self,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> DiskPromptCacheRestorePlan | None:
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
    ) -> _HotRestoreCandidate | None:
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

        return _HotRestoreCandidate(
            entry=entry,
            cached_prefix_len=target_prefix_len,
        )

    def _load_hot_restore_plan(
        self,
        plan: _HotRestoreCandidate,
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
        disk_plan: DiskPromptCacheRestorePlan,
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

    def save_prompt_cache_snapshot(
        self,
        prompt_cache: list[Any],
        prefix_chunks: list[PromptPrefixChunk],
        start_chunk_idx: int,
        end_chunk_idx: int,
        snapshot_len: int,
    ) -> None:
        """Prepare and enqueue crossed prompt-cache chunks from one snapshot."""
        if not self._cache_store.can_store_records():
            return

        for chunk_idx in range(start_chunk_idx, end_chunk_idx):
            chunk = prefix_chunks[chunk_idx]
            try:
                pending_save = self._cache_store.prepare_save(
                    chunk=chunk,
                    prefix_chunks=prefix_chunks[: chunk_idx + 1],
                    prompt_cache=prompt_cache,
                    # Opaque state caches are only exact at this model-call end.
                    save_state_checkpoint=chunk.end == snapshot_len,
                )
                self._enqueue_pending_save(pending_save)
            except PromptCacheRecordCoverageError as exc:
                logger.warning(
                    "Skipping prompt cache save for chunk [%s, %s) at snapshot %s: %s",
                    chunk.start,
                    chunk.end,
                    snapshot_len,
                    exc,
                )
                continue
            except Exception:
                logger.debug(
                    "Skipping prompt cache save for chunk [%s, %s) at snapshot %s.",
                    chunk.start,
                    chunk.end,
                    snapshot_len,
                    exc_info=True,
                )
                continue

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
