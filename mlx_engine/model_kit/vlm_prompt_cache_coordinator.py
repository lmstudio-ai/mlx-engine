import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mlx_engine.model_kit.vlm_prompt_cache_types import (
    PendingPromptCacheSave,
    PromptImageSpan,
    build_prefix_cache_boundaries,
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.vlm_prompt_spill_cache import VlmPromptSpillCache

logger = logging.getLogger(__name__)


@dataclass
class RestoredPromptCache:
    cached_prefix_len: int
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

    def restore(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> RestoredPromptCache | None:
        if len(prompt_input_ids) == 0:
            return None

        prefix_match = self._spill_cache.find_longest_prefix(
            prompt_input_ids,
            image_spans,
        )
        if prefix_match is None:
            return None

        try:
            cached_state = self._spill_cache.load_chunk_sequence(
                prefix_match.chunk_keys
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
            cached_prefix_len=prefix_match.matched_prefix_len,
            prompt_cache=cached_state.prompt_cache,
            rope_deltas=cached_state.rope_deltas,
        )

    def boundaries_after(
        self,
        *,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
        prompt_progress: int,
    ) -> list[int]:
        return [
            boundary
            for boundary in build_prefix_cache_boundaries(
                prompt_input_ids,
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
        chunks_by_end = {
            chunk.end: chunk
            for chunk in build_prefix_cache_chunks(
                full_prompt_input_ids,
                image_spans,
            )
        }

        def _callback(
            _uid: int,
            prefix_len: int,
            prompt_cache: list[Any],
            decode_state: Optional[dict[str, Any]],
        ) -> None:
            chunk = chunks_by_end.get(prefix_len)
            if prefix_len <= 0 or chunk is None:
                return

            rope_deltas = (
                None if decode_state is None else decode_state.get("rope_deltas")
            )
            # V1 prefill batches are single-sequence. Persist the per-request
            # cache shape, not mlx-lm's transient batched cache wrapper.
            prompt_cache_for_save = [
                cache.extract(0) if hasattr(cache, "extract") else cache
                for cache in prompt_cache
            ]
            pending_save = self._spill_cache.prepare_save(
                chunk=chunk,
                prompt_cache=prompt_cache_for_save,
                rope_deltas=rope_deltas,
            )
            if pending_save is not None:
                self._enqueue_pending_save(pending_save)

        return _callback
