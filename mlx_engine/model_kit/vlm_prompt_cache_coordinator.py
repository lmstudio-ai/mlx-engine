import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

from mlx_engine.model_kit.vlm_prompt_cache_types import (
    PendingPromptCacheSave,
    PromptImageSpan,
    build_prefix_cache_chunks,
    build_prefix_cache_save_points_for_length,
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
    prompt_input_ids: list[int]
    image_spans: list[PromptImageSpan]
    prompt_cache: list[Any]
    rope_deltas: Optional[Any]


class VlmPromptCacheCoordinator:
    def __init__(
        self,
        spill_cache: VlmPromptSpillCache,
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
        if len(prompt_input_ids) == 0:
            return None

        hot_restored = self._restore_hot_entry(
            prompt_input_ids=prompt_input_ids,
            image_spans=image_spans,
        )
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
            rope_deltas=cached_state.rope_deltas,
        )

    def boundaries_after(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
        prompt_progress: int,
        max_prefix_len: int | None = None,
    ) -> list[int]:
        planned_prefix_len = max_prefix_len or len(prompt_input_ids)
        return [
            boundary
            for boundary in build_prefix_cache_save_points_for_length(
                planned_prefix_len,
                image_spans,
            )
            if boundary > prompt_progress
        ]

    def make_boundary_callback(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ):
        full_prompt_input_ids = list(prompt_input_ids)
        image_spans = list(image_spans)

        def _callback(
            _uid: int,
            prefix_len: int,
            prompt_cache: list[Any],
            decode_state: Optional[dict[str, Any]],
            prompt_input_ids: Optional[list[int]] = None,
        ) -> None:
            if not self._spill_cache.can_store_records():
                return

            snapshot_input_ids = (
                full_prompt_input_ids
                if prompt_input_ids is None
                else list(prompt_input_ids)
            )
            chunks_by_end = {
                chunk.end: chunk
                for chunk in build_prefix_cache_chunks(
                    snapshot_input_ids,
                    image_spans,
                )
            }
            chunk = chunks_by_end.get(prefix_len)
            if prefix_len <= 0 or chunk is None:
                return

            rope_deltas = (
                None if decode_state is None else decode_state.get("rope_deltas")
            )
            pending_save = self._spill_cache.prepare_save(
                chunk=chunk,
                prompt_cache=prompt_cache,
                rope_deltas=rope_deltas,
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

            common_prefix_len = _image_safe_common_prefix_len(
                prompt_input_ids,
                image_spans,
                entry.prompt_input_ids,
                entry.image_spans,
            )
            target_prefix_len = min(common_prefix_len, len(prompt_input_ids) - 1)
            if target_prefix_len <= 0:
                return None

            cached_prefix_len = len(entry.prompt_input_ids)
            trim_count = cached_prefix_len - target_prefix_len
            if trim_count < 0:
                return None
            if trim_count > 0:
                if entry.rope_deltas is not None:
                    return None
                if not can_trim_prompt_cache(entry.prompt_cache):
                    return None

            # The hot cache is mutable runtime state; restore consumes it.
            self._hot_entry = None

        if trim_count > 0:
            trimmed = trim_prompt_cache(entry.prompt_cache, trim_count)
            if trimmed != trim_count:
                return None
            entry.prompt_input_ids = entry.prompt_input_ids[:target_prefix_len]
            entry.image_spans = [
                span for span in entry.image_spans if span.end <= target_prefix_len
            ]

        return RestoredPromptCache(
            cached_prefix_len=target_prefix_len,
            prompt_cache=entry.prompt_cache,
            rope_deltas=entry.rope_deltas,
        )


def _image_safe_common_prefix_len(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    cached_input_ids: list[int],
    cached_image_spans: list[PromptImageSpan],
) -> int:
    common_prefix_len = 0
    for token, cached_token in zip(prompt_input_ids, cached_input_ids):
        if token != cached_token:
            break
        common_prefix_len += 1

    while True:
        safe_prefix_len = _shrink_prefix_before_unmatched_image(
            common_prefix_len,
            image_spans,
            cached_image_spans,
        )
        safe_prefix_len = _shrink_prefix_before_unmatched_image(
            safe_prefix_len,
            cached_image_spans,
            image_spans,
        )
        if safe_prefix_len == common_prefix_len:
            return common_prefix_len
        common_prefix_len = safe_prefix_len


def _shrink_prefix_before_unmatched_image(
    prefix_len: int,
    spans: list[PromptImageSpan],
    other_spans: list[PromptImageSpan],
) -> int:
    for span in spans:
        if span.start >= prefix_len:
            continue
        if span.end > prefix_len:
            return span.start
        if not any(
            other.start == span.start
            and other.end == span.end
            and other.image_hash == span.image_hash
            for other in other_spans
        ):
            return span.start
    return prefix_len
