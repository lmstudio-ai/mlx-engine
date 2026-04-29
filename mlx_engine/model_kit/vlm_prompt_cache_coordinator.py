import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

from mlx_engine.model_kit.vlm_prompt_cache_types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PendingPromptCacheSave,
    PromptImageSpan,
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.vlm_prompt_spill_cache import VlmPromptSpillCache
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

logger = logging.getLogger(__name__)


@dataclass
class RestoredPromptCache:
    cached_prefix_len: int
    prompt_cache: list[Any]
    rope_deltas: Optional[Any]


@dataclass
class _HotPromptCacheEntry:
    """One completed mutable prompt cache kept in memory for fast follow-up."""

    prompt_input_ids: list[int]
    image_spans: list[PromptImageSpan]
    prompt_cache: list[Any]
    rope_deltas: Optional[Any]


class VlmPromptCacheCoordinator:
    """Coordinates hot-cache reuse and actor-owned disk spill saves.

    The scheduler thread records the most recently completed runtime cache.
    Request preparation may consume that hot cache before falling back to disk.
    """

    def __init__(
        self,
        spill_cache: VlmPromptSpillCache,
        # Called after the scheduler thread prepares an immutable save payload.
        # The owner decides how to persist it, usually via actor disk I/O.
        enqueue_pending_save: Callable[[PendingPromptCacheSave], None],
    ):
        self._spill_cache = spill_cache
        self._enqueue_pending_save = enqueue_pending_save
        self._hot_entry_lock = Lock()
        self._hot_entry: _HotPromptCacheEntry | None = None

    def restore(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> RestoredPromptCache | None:
        try:
            hot_restored = self._restore_hot_entry(
                prompt_input_ids=prompt_input_ids,
                image_spans=image_spans,
            )
        except Exception:
            # Hot restore is an optimization; generation can recompute on miss.
            logger.debug(
                "Hot prompt cache restore failed; treating it as a cache miss.",
                exc_info=True,
            )
            hot_restored = None
        if hot_restored is not None:
            return hot_restored

        try:
            cached_state = self._spill_cache.restore_longest_prefix(
                prompt_input_ids,
                image_spans,
            )
        except Exception:
            # Spill restore is an optimization; generation can recompute on miss.
            logger.debug(
                "Prompt spill cache restore failed; treating it as a cache miss.",
                exc_info=True,
            )
            return None
        if cached_state is None:
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
    ):
        image_spans = list(image_spans)

        def _callback(
            prompt_cache: list[Any],
            snapshot_input_ids: list[int],
        ) -> None:
            if not self._spill_cache.can_store_records():
                return
            snapshot_input_ids = list(snapshot_input_ids)
            chunks = build_prefix_cache_chunks(
                snapshot_input_ids,
                image_spans,
            )
            if not chunks:
                return

            pending_save = self._spill_cache.prepare_save(
                chunk=chunks[-1],
                prefix_chunks=chunks,
                prompt_cache=prompt_cache,
            )
            self._enqueue_pending_save(pending_save)

        return _callback

    def remember_completed(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
        prompt_cache: list[Any],
        rope_deltas: Optional[Any],
    ) -> None:
        """Keep exactly one completed cache hot for the next likely follow-up."""
        with self._hot_entry_lock:
            self._hot_entry = _HotPromptCacheEntry(
                prompt_input_ids=list(prompt_input_ids),
                image_spans=list(image_spans),
                prompt_cache=prompt_cache,
                rope_deltas=rope_deltas,
            )

    def _restore_hot_entry(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> RestoredPromptCache | None:
        with self._hot_entry_lock:
            entry = self._hot_entry
            if entry is None:
                return None
            # A new request makes the old hot cache cold; do not keep it while
            # this request recomputes or restores from disk.
            self._hot_entry = None

        target_prefix_len = _image_safe_common_prefix_len(
            prompt_input_ids,
            image_spans,
            entry.prompt_input_ids,
            entry.image_spans,
            max_prefix_len=len(prompt_input_ids) - 1,
        )
        if target_prefix_len <= 0:
            return None

        trim_count = len(entry.prompt_input_ids) - target_prefix_len
        if trim_count > 0:
            if not can_trim_prompt_cache(entry.prompt_cache):
                return None
            trimmed = trim_prompt_cache(entry.prompt_cache, trim_count)
            if trimmed != trim_count:
                return None

        return RestoredPromptCache(
            cached_prefix_len=target_prefix_len,
            prompt_cache=entry.prompt_cache,
            rope_deltas=None if trim_count > 0 else entry.rope_deltas,
        )


def _image_safe_common_prefix_len(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    cached_input_ids: list[int],
    cached_image_spans: list[PromptImageSpan],
    *,
    max_prefix_len: int,
) -> int:
    """Return the token prefix that does not split or mismatch image spans."""
    common_prefix_len = 0
    for token, cached_token in zip(prompt_input_ids, cached_input_ids):
        if token != cached_token:
            break
        common_prefix_len += 1
    common_prefix_len = min(common_prefix_len, max_prefix_len)

    # Placeholder token ids can match while the underlying image differs.
    image_keys = {_image_span_key(span) for span in image_spans}
    cached_image_keys = {_image_span_key(span) for span in cached_image_spans}
    # Either prompt may contain a non-reusable image inside the prefix.
    non_reusable_starts = [
        # New prompt images must match images already present in the hot cache.
        _first_non_reusable_image_start(
            common_prefix_len,
            image_spans,
            cached_image_keys,
        ),
        # Hot-cache images must also still exist in the new prompt.
        _first_non_reusable_image_start(
            common_prefix_len,
            cached_image_spans,
            image_keys,
        ),
    ]
    non_reusable_starts = [start for start in non_reusable_starts if start is not None]
    return min(non_reusable_starts) if non_reusable_starts else common_prefix_len


def _image_span_key(span: PromptImageSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.image_hash


def _first_non_reusable_image_start(
    prefix_len: int,
    spans: list[PromptImageSpan],
    other_span_keys: set[tuple[int, int, str]],
) -> int | None:
    for span in spans:
        if span.start >= prefix_len:
            break
        if span.end > prefix_len:
            return span.start
        if _image_span_key(span) not in other_span_keys:
            return span.start
    return None
