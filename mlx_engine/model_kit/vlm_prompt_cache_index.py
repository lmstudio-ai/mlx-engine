from collections.abc import Callable
from typing import Optional

from mlx_engine.model_kit.vlm_prompt_cache_types import (
    PromptCacheChunkMetadata,
    PromptCacheRecordMetadata,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    make_record_key,
)


class PromptCacheIndexView:
    """Read-only policy view over the prompt-cache indexes.

    Restore policy:
    - KV deltas are needed for every chunk in the prefix chain.
    - Rotating deltas are needed only inside the target sliding window.
    - Opaque state checkpoints are needed only at the exact target chunk.

    Short-lived by design: callers construct this from the actor-owned spill
    cache indexes when they need to evaluate restore availability.
    """

    def __init__(
        self,
        *,
        metadata_by_key: dict[str, PromptCacheChunkMetadata],
        record_metadata_by_key: dict[str, PromptCacheRecordMetadata],
        record_exists: Callable[[str], bool],
    ):
        self._metadata_by_key = metadata_by_key
        self._record_metadata_by_key = record_metadata_by_key
        self._record_exists = record_exists

    def restore_record_keys_for_chunk_chain(
        self, chunk_keys: list[str]
    ) -> Optional[dict[str, list[str]]]:
        """Return physical records needed to restore a cached chunk chain.

        Returns None when the index says the chain is not currently restorable:
        metadata is missing, required records are not indexed, or blobs were
        already evicted from the spool.
        """
        if not chunk_keys:
            return {}

        # The last chunk is the restore boundary for SWA windowing/checkpoints.
        target_chunk_key = chunk_keys[-1]
        target_metadata = self._metadata_by_key.get(target_chunk_key)
        if target_metadata is None:
            return None

        target_chunk_end = target_metadata.chunk_end
        rotating_window_size = self._max_rotating_window_size_for_chain(chunk_keys)
        record_keys_by_chunk_key: dict[str, list[str]] = {}
        for chunk_key in chunk_keys:
            chunk_metadata = self._metadata_by_key.get(chunk_key)
            if chunk_metadata is None:
                return None

            record_keys: list[str] = []
            for record_kind in RECORD_WRITE_ORDER:
                if record_kind not in chunk_metadata.payload_kinds:
                    continue
                if record_kind == RECORD_KIND_STATE_CHECKPOINT:
                    # Opaque state caches are exact-boundary checkpoints.
                    if chunk_key != target_chunk_key:
                        continue
                elif record_kind == RECORD_KIND_ROTATING_DELTA:
                    if rotating_window_size is None:
                        return None
                    if not self._rotating_chunk_overlaps_target_window(
                        chunk_metadata=chunk_metadata,
                        target_chunk_end=target_chunk_end,
                        rotating_window_size=rotating_window_size,
                    ):
                        continue

                record_key = make_record_key(chunk_key, record_kind)
                if not self._record_exists_in_index_and_store(record_key):
                    return None
                record_keys.append(record_key)
            record_keys_by_chunk_key[chunk_key] = record_keys

        return record_keys_by_chunk_key

    def _record_exists_in_index_and_store(self, record_key: str) -> bool:
        return record_key in self._record_metadata_by_key and self._record_exists(
            record_key
        )

    def _rotating_chunk_overlaps_target_window(
        self,
        *,
        chunk_metadata: PromptCacheChunkMetadata,
        target_chunk_end: int,
        rotating_window_size: int,
    ) -> bool:
        # Sliding-window layers only need chunks overlapping the target boundary.
        window_start = target_chunk_end - rotating_window_size
        return chunk_metadata.chunk_end > window_start

    def _max_rotating_window_size_for_chain(
        self, chunk_keys: list[str]
    ) -> Optional[int]:
        """Return the widest rotating window needed by this chunk chain."""
        window_size = None
        for chunk_key in chunk_keys:
            record_key = make_record_key(chunk_key, RECORD_KIND_ROTATING_DELTA)
            record_metadata = self._record_metadata_by_key.get(record_key)
            if record_metadata is not None and record_metadata.window_size is not None:
                window_size = max(window_size or 0, record_metadata.window_size)

        return window_size
