from collections.abc import Callable
from typing import Optional

from mlx_engine.model_kit.vlm_prompt_cache_payload import (
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
)
from mlx_engine.model_kit.vlm_prompt_cache_types import (
    CachedPromptMetadata,
    CachedPromptRecordMetadata,
    build_prefix_cache_chunks,
    make_record_key,
)


class PromptCacheDependencyPlanner:
    """Selects records needed to restore or safely evict prompt chunks.

    The spill cache owns synchronization; callers should hold its metadata lock.
    """

    def __init__(
        self,
        *,
        metadata_by_key: dict[str, CachedPromptMetadata],
        record_metadata_by_key: dict[str, CachedPromptRecordMetadata],
        record_exists: Callable[[str], bool],
    ):
        self._metadata_by_key = metadata_by_key
        self._record_metadata_by_key = record_metadata_by_key
        self._record_exists = record_exists

    def record_keys_for_chunk_sequence(
        self, chunk_keys: list[str]
    ) -> Optional[dict[str, list[str]]]:
        if not chunk_keys:
            return {}

        final_metadata = self._metadata_by_key.get(chunk_keys[-1])
        if final_metadata is None:
            return None

        final_chunk_end = final_metadata.chunk_end
        final_chunk_key = chunk_keys[-1]
        rotating_window_size = self._rotating_window_size_for_sequence(chunk_keys)
        record_keys_by_chunk_key = {}
        for chunk_key in chunk_keys:
            chunk_metadata = self._metadata_by_key.get(chunk_key)
            if chunk_metadata is None:
                return None

            record_keys = []
            for record_kind in RECORD_WRITE_ORDER:
                if record_kind not in chunk_metadata.payload_kinds:
                    continue
                if (
                    record_kind == RECORD_KIND_STATE_CHECKPOINT
                    and chunk_key != final_chunk_key
                ):
                    continue
                if record_kind == RECORD_KIND_ROTATING_DELTA:
                    if rotating_window_size is None:
                        return None
                    window_start = final_chunk_end - rotating_window_size
                    if chunk_metadata.chunk_end <= window_start:
                        continue

                record_key = make_record_key(chunk_key, record_kind)
                if (
                    record_key not in self._record_metadata_by_key
                    or not self._record_exists(record_key)
                ):
                    return None
                record_keys.append(record_key)
            record_keys_by_chunk_key[chunk_key] = record_keys

        return record_keys_by_chunk_key

    def chunk_record_keys(
        self, chunk_key: str, metadata: CachedPromptMetadata
    ) -> list[str]:
        return [
            make_record_key(chunk_key, record_kind)
            for record_kind in RECORD_WRITE_ORDER
            if record_kind in metadata.payload_kinds
        ]

    def chunk_has_live_records(self, chunk_key: str) -> bool:
        metadata = self._metadata_by_key.get(chunk_key)
        if metadata is None:
            return False

        return any(
            record_key in self._record_metadata_by_key
            for record_key in self.chunk_record_keys(chunk_key, metadata)
        )

    def has_dependent_chunks(
        self, chunk_key: str, metadata: CachedPromptMetadata
    ) -> bool:
        for candidate_key, candidate_metadata in self._metadata_by_key.items():
            if (
                candidate_key == chunk_key
                or candidate_metadata.chunk_end <= metadata.chunk_end
                or candidate_metadata.image_hashes != metadata.image_hashes
                or not self.chunk_has_live_records(candidate_key)
            ):
                continue

            if (
                candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                == metadata.prompt_input_ids
            ):
                return True

        return False

    def is_stale_optional_record(
        self,
        record_metadata: CachedPromptRecordMetadata,
        metadata: CachedPromptMetadata,
    ) -> bool:
        if record_metadata.record_kind == RECORD_KIND_ROTATING_DELTA:
            return self._is_stale_rotating_record(record_metadata, metadata)
        if record_metadata.record_kind == RECORD_KIND_STATE_CHECKPOINT:
            return self._is_stale_state_checkpoint_record(record_metadata, metadata)
        return False

    def _rotating_window_size_for_sequence(
        self, chunk_keys: list[str]
    ) -> Optional[int]:
        window_size = None
        for chunk_key in chunk_keys:
            record_key = make_record_key(chunk_key, RECORD_KIND_ROTATING_DELTA)
            record_metadata = self._record_metadata_by_key.get(record_key)
            if record_metadata is not None and record_metadata.window_size is not None:
                window_size = max(window_size or 0, record_metadata.window_size)

        return window_size

    def _is_stale_rotating_record(
        self,
        record_metadata: CachedPromptRecordMetadata,
        metadata: CachedPromptMetadata,
    ) -> bool:
        if record_metadata.window_size is None:
            return False

        for candidate_key, candidate_metadata in self._metadata_by_key.items():
            if (
                candidate_key == record_metadata.chunk_key
                or candidate_metadata.chunk_end <= metadata.chunk_end
                or candidate_metadata.image_hashes != metadata.image_hashes
                or not self.chunk_has_live_records(candidate_key)
                or candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                != metadata.prompt_input_ids
            ):
                continue

            window_start = candidate_metadata.chunk_end - record_metadata.window_size
            if metadata.chunk_end > window_start:
                continue

            if self._candidate_sequence_loadable(candidate_key, candidate_metadata):
                return True

        return False

    def _is_stale_state_checkpoint_record(
        self,
        record_metadata: CachedPromptRecordMetadata,
        metadata: CachedPromptMetadata,
    ) -> bool:
        for candidate_key, candidate_metadata in self._metadata_by_key.items():
            if (
                candidate_key == record_metadata.chunk_key
                or candidate_metadata.chunk_end <= metadata.chunk_end
                or candidate_metadata.image_hashes != metadata.image_hashes
                or not self.chunk_has_live_records(candidate_key)
                or candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                != metadata.prompt_input_ids
            ):
                continue

            if self._candidate_sequence_loadable(candidate_key, candidate_metadata):
                return True

        return False

    def _candidate_sequence_loadable(
        self,
        candidate_key: str,
        candidate_metadata: CachedPromptMetadata,
    ) -> bool:
        candidate_chunk_keys = [
            chunk.key
            for chunk in build_prefix_cache_chunks(
                candidate_metadata.prompt_input_ids,
                candidate_metadata.image_hashes,
                candidate_metadata.min_reusable_prefix_len,
            )
            if chunk.end <= candidate_metadata.chunk_end
        ]
        return bool(
            candidate_chunk_keys
            and candidate_chunk_keys[-1] == candidate_key
            and self.record_keys_for_chunk_sequence(candidate_chunk_keys) is not None
        )
