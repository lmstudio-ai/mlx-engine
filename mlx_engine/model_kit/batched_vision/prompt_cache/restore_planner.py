from collections.abc import Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptCacheLayout,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    make_record_key,
)


class PromptCacheRestorePlanner:
    """Read-only planner for records needed to restore a prompt prefix.

    Restore policy:
    - KV deltas are needed for every chunk in the prefix chain.
    - Rotating deltas are needed only inside the target sliding window.
    - Opaque state checkpoints are needed only at the exact target chunk.

    Short-lived by design: callers construct this from the cache-I/O-thread-owned
    physical record index when they need to evaluate restore availability.
    """

    def __init__(
        self,
        *,
        layout: PromptCacheLayout,
        record_metadata_by_key: dict[str, PromptCacheRecordMetadata],
        record_exists: Callable[[str], bool],
    ):
        self._layout = layout
        self._record_metadata_by_key = record_metadata_by_key
        self._record_exists = record_exists

    def restore_record_keys_for_chunk_chain(
        self, chunks: list[PromptPrefixChunk]
    ) -> dict[str, list[str]] | None:
        """Return physical records needed to restore a cached chunk chain.

        Returns None when the index says the chain is not currently restorable:
        required records are not indexed, or blobs were already evicted from the
        blob store.
        """
        if not chunks:
            return {}

        # The last chunk is the restore boundary for SWA windowing/checkpoints.
        target_chunk = chunks[-1]
        target_chunk_end = target_chunk.end
        rotating_window_size = self._layout.rotating_window_size
        record_keys_by_chunk_key: dict[str, list[str]] = {}
        for chunk in chunks:
            record_keys: list[str] = []
            for record_kind in RECORD_WRITE_ORDER:
                if not self._layout.layer_indices_by_kind.get(record_kind):
                    continue
                if record_kind == RECORD_KIND_STATE_CHECKPOINT:
                    # Opaque state caches are exact-boundary checkpoints.
                    if chunk.key != target_chunk.key:
                        continue
                elif record_kind == RECORD_KIND_ROTATING_DELTA:
                    if rotating_window_size is None:
                        return None
                    if not self._rotating_chunk_overlaps_target_window(
                        chunk=chunk,
                        target_chunk_end=target_chunk_end,
                        rotating_window_size=rotating_window_size,
                    ):
                        continue

                record_key = make_record_key(chunk.key, record_kind)
                if not self._has_record(record_key):
                    return None
                record_keys.append(record_key)
            record_keys_by_chunk_key[chunk.key] = record_keys

        return record_keys_by_chunk_key

    def _has_record(self, record_key: str) -> bool:
        return record_key in self._record_metadata_by_key and self._record_exists(
            record_key
        )

    def _rotating_chunk_overlaps_target_window(
        self,
        *,
        chunk: PromptPrefixChunk,
        target_chunk_end: int,
        rotating_window_size: int,
    ) -> bool:
        # Sliding-window layers only need chunks overlapping the target boundary.
        window_start = target_chunk_end - rotating_window_size
        return chunk.end > window_start
