from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path
from time import monotonic
from typing import Any

import mlx.core as mx
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.disk_budget import (
    final_cache_store_budget_bytes,
    provisional_cache_store_budget_bytes,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    assemble_prompt_cache_chunks,
    make_prompt_cache_layout,
    prepare_prompt_cache_records_for_chunk,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.restore_planner import (
    PromptCacheRestorePlanner,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptCacheLayout,
    PromptImageSpan,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    PreparedPromptRecord,
    PendingPromptCacheSave,
    PromptCacheStoreStats,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    RecordKind,
    LoadedDiskPromptCache,
    make_record_key,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.blob_store import (
    TemporarySafetensorBlobStore,
)
from mlx.utils import tree_flatten


logger = logging.getLogger(__name__)


_MIB_BYTES = 1024 * 1024
_CACHE_USAGE_LOG_COOLDOWN_SECONDS = 60.0
_RECORD_RETENTION_PRIORITY: tuple[RecordKind, ...] = (
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
)


@dataclass
class DiskPromptCacheRestorePlan:
    """Same-thread selection of disk records for one prompt-cache restore.

    The caller may compare this with a hot-cache candidate, but any selected
    plan must be loaded without yielding off the cache I/O thread.
    """

    cached_prefix_len: int
    chunks: list[PromptPrefixChunk]
    record_keys_by_chunk_key: dict[str, list[str]]


def _prefix_len_splits_image_span(
    prefix_len: int,
    image_spans: list[PromptImageSpan],
) -> bool:
    # A chunk ending inside an image span can be an internal record written from
    # a later full visual-prefill snapshot, but it is not a valid terminal restore
    # point because the cached image-token states depended on later image tokens.
    return any(span.start < prefix_len < span.end for span in image_spans)


class VlmPromptCacheStore:
    """Cache-I/O-thread-owned index and temporary safetensor blob store.

    Mutable index/blob-store operations run on the prompt-cache I/O thread. The
    generation thread may prepare immutable cache records and make advisory
    budget checks, but it must not mutate committed cache store state.

    Invariants:
    - A selected restore chain is loaded without interleaved eviction.
    - Selected record keys exist in both metadata and the blob store.
    - Touched LRU keys are committed physical records.
    """

    def __init__(self, max_kv_size: int | None = None):
        base_dir = Path("/tmp")
        self._base_dir = base_dir
        self._max_kv_size = max_kv_size
        self._blob_store = TemporarySafetensorBlobStore(base_dir)
        self._empirical_budget_set = False
        self._layout: PromptCacheLayout | None = None
        self._record_metadata_by_key: dict[str, PromptCacheRecordMetadata] = {}
        self._key_sizes: dict[str, int] = {}
        self._lru_keys: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._max_cache_store_bytes = provisional_cache_store_budget_bytes(base_dir)
        self._restore_hit_tokens = 0
        self._restore_miss_tokens = 0
        self._cache_evictions = 0
        self._cache_evicted_bytes = 0
        self._last_cache_usage_log_time = 0.0
        logger.info(
            "VLM prompt cache disk store: lifetime=model_load storage=temporary "
            "cleanup=model_unload_or_process_exit"
        )

    def plan_longest_prefix_restore(
        self,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> DiskPromptCacheRestorePlan | None:
        """Select the longest matching cacheable prefix without loading blobs.

        The final prompt token stays uncached so generation has a suffix to
        process. Too-short or unchunkable prompts do not produce a disk plan.
        """
        max_reusable_prefix_len = len(prompt_input_ids) - 1
        if max_reusable_prefix_len <= 0:
            return None
        if self._layout is None:
            return None

        eligible_chunks = [
            chunk
            for chunk in build_prefix_cache_chunks(prompt_input_ids, image_spans)
            if chunk.end <= max_reusable_prefix_len
        ]
        if not eligible_chunks:
            return None

        restore_planner = self._cache_restore_planner()
        # A later SWA boundary may be restorable even if an earlier boundary is
        # missing old rotating records, so scan downward for the best boundary.
        for end_idx in range(len(eligible_chunks), 0, -1):
            chunks = eligible_chunks[:end_idx]
            if _prefix_len_splits_image_span(chunks[-1].end, image_spans):
                continue
            record_keys_by_chunk_key = (
                restore_planner.restore_record_keys_for_chunk_chain(chunks)
            )
            if record_keys_by_chunk_key is not None:
                return DiskPromptCacheRestorePlan(
                    cached_prefix_len=chunks[-1].end,
                    chunks=chunks,
                    record_keys_by_chunk_key=record_keys_by_chunk_key,
                )

        return None

    def load_restore_plan(
        self,
        plan: DiskPromptCacheRestorePlan,
    ) -> LoadedDiskPromptCache:
        """Load a restore plan selected on this cache I/O thread."""
        layout = self._require_layout()

        chunk_prompt_caches = []

        # Load each chunk's physical records into sparse per-layer cache lists.
        for chunk in plan.chunks:
            prompt_cache = self._load_one_chunk(
                plan.record_keys_by_chunk_key[chunk.key],
                layout,
            )
            chunk_prompt_caches.append(prompt_cache)

        prompt_cache = assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            plan.chunks,
            layout,
        )
        # Disk restores run on the prompt-cache I/O thread; decode consumes the
        # cache on the generation thread. Force assembled arrays now so no lazy
        # graph keeps a thread-local MLX stream from the restore worker.
        mx.eval(
            [
                value
                for _, value in tree_flatten([cache.state for cache in prompt_cache])
            ]
        )

        # Restore access refreshes exactly the records used by this chain.
        for record_key in self._ordered_record_keys_for_touch(
            [chunk.key for chunk in plan.chunks],
            plan.record_keys_by_chunk_key,
        ):
            self._touch_cache_entry(record_key)

        return LoadedDiskPromptCache(
            cached_prefix_len=plan.cached_prefix_len,
            prompt_cache=prompt_cache,
        )

    def record_restore_tokens(
        self,
        *,
        hit_tokens: int,
        miss_tokens: int,
    ) -> tuple[int, int]:
        """Record one request and return lifetime hit/miss token totals."""
        self._restore_hit_tokens += hit_tokens
        self._restore_miss_tokens += miss_tokens
        return self._restore_hit_tokens, self._restore_miss_tokens

    def _load_one_chunk(
        self,
        record_keys: list[str],
        layout: PromptCacheLayout,
    ) -> list[Any]:
        prompt_cache: list[Any] = [None] * len(layout.layer_kinds)
        for record_key in record_keys:
            try:
                record_prompt_cache = self._blob_store.load_record(record_key)
            except Exception:
                self._evict_key(record_key)
                raise
            record_metadata = self._record_metadata_by_key[record_key]

            for layer_idx, cache in zip(
                record_metadata.layer_indices, record_prompt_cache
            ):
                prompt_cache[layer_idx] = cache

        return prompt_cache

    def can_store_records(self) -> bool:
        """Return False when the cache store is intentionally hot-only."""
        return self._max_cache_store_bytes != 0

    def prepare_save(
        self,
        *,
        chunk: PromptPrefixChunk,
        prefix_chunks: list[PromptPrefixChunk],
        prompt_cache: list[Any],
        save_state_checkpoint: bool = True,
    ) -> PendingPromptCacheSave:
        """Prepare a cache save for the cache I/O thread."""
        record_caches, record_kinds = prepare_prompt_cache_records_for_chunk(
            prompt_cache,
            chunk.start,
            chunk.end,
        )
        layout = make_prompt_cache_layout(record_caches, record_kinds)
        records = []
        for record_kind in RECORD_WRITE_ORDER:
            if (
                record_kind == RECORD_KIND_STATE_CHECKPOINT
                and not save_state_checkpoint
            ):
                continue
            layer_indices = layout.layer_indices_by_kind.get(record_kind, [])
            if not layer_indices:
                continue

            records.append(
                self._prepare_record_save(
                    chunk_key=chunk.key,
                    record_kind=record_kind,
                    layer_indices=layer_indices,
                    record_cache=[record_caches[idx] for idx in layer_indices],
                )
            )

        return PendingPromptCacheSave(
            prefix_chunks=prefix_chunks,
            cache_layout=layout,
            records=records,
        )

    def budget_update_from_completed_cache(self, prompt_cache: list[Any]) -> int | None:
        """Return the empirical cache store budget from a completed cache."""
        if self._empirical_budget_set:
            return None

        try:
            return final_cache_store_budget_bytes(
                self._base_dir,
                prompt_cache,
                self._max_kv_size,
            )
        except Exception:
            logger.warning(
                "Failed to estimate VLM prompt cache disk budget; "
                "disabling disk records",
                exc_info=True,
            )
            return 0

    def commit_budget_update(self, max_cache_store_bytes: int) -> None:
        """Set the empirical budget and evict records from the cache I/O thread."""
        if self._empirical_budget_set:
            return
        self._max_cache_store_bytes = max_cache_store_bytes
        self._empirical_budget_set = True

        self._evict_if_needed()

    def commit_pending_save(self, pending_save: PendingPromptCacheSave) -> None:
        """Commit a pending save from the cache I/O thread."""
        if not self.can_store_records():
            return
        if self._layout is None:
            self._layout = pending_save.cache_layout

        try:
            for record in pending_save.records:
                if self._blob_store.exists(record.key):
                    self._record_metadata_by_key[record.key] = record.metadata
                    self._touch_cache_entry(record.key)
                    continue

                # The I/O thread waits, writes, then publishes/account each record.
                mx.eval(list(record.snapshot_arrays.values()))
                self._blob_store.put(
                    record.key,
                    record.snapshot_arrays,
                    record.safetensor_metadata,
                )
                self._record_metadata_by_key[record.key] = record.metadata
                self._touch_cache_entry(record.key)

        finally:
            self._touch_longest_budget_fit_restore_chain(pending_save.prefix_chunks)
            self._evict_if_needed()
            self._maybe_log_cache_usage()

    def snapshot_stats(self) -> PromptCacheStoreStats:
        """Return best-effort diagnostics for smokes/debug output."""
        total_bytes = self._total_bytes
        max_bytes = self._max_cache_store_bytes
        entry_count = len(self._record_metadata_by_key)
        hit_tokens = self._restore_hit_tokens
        miss_tokens = self._restore_miss_tokens
        evictions = self._cache_evictions
        record_sizes_by_key = dict(self._key_sizes)
        record_metadata_by_key = dict(self._record_metadata_by_key)
        chunk_sizes_by_key = {}
        chunk_records_available_by_key = {}
        chunk_keys = sorted(
            {metadata.chunk_key for metadata in record_metadata_by_key.values()}
        )
        for chunk_key in chunk_keys:
            record_keys = [
                record_key
                for record_key, metadata in record_metadata_by_key.items()
                if metadata.chunk_key == chunk_key
            ]
            chunk_sizes_by_key[chunk_key] = sum(
                record_sizes_by_key.get(record_key, 0) for record_key in record_keys
            )
            chunk_records_available_by_key[chunk_key] = bool(record_keys) and all(
                record_key in record_sizes_by_key
                and self._blob_store.exists(record_key)
                for record_key in record_keys
            )

        return PromptCacheStoreStats(
            total_bytes=total_bytes,
            max_bytes=max_bytes,
            entry_count=entry_count,
            hit_tokens=hit_tokens,
            miss_tokens=miss_tokens,
            evictions=evictions,
            record_sizes=sorted(record_sizes_by_key.values()),
            record_sizes_by_key=record_sizes_by_key,
            chunk_sizes_by_key=chunk_sizes_by_key,
            chunk_records_available_by_key=chunk_records_available_by_key,
        )

    def close(self) -> None:
        """Clear metadata and close the temporary blob store."""
        self._layout = None
        self._record_metadata_by_key.clear()
        self._key_sizes.clear()
        self._lru_keys.clear()
        self._total_bytes = 0
        self._blob_store.close()

    def _prepare_record_save(
        self,
        *,
        chunk_key: str,
        record_kind: RecordKind,
        layer_indices: list[int],
        record_cache: list[Any],
    ) -> PreparedPromptRecord:
        record_key = make_record_key(chunk_key, record_kind)
        cache_data = [cache.state for cache in record_cache]
        cache_meta_states = [cache.meta_state for cache in record_cache]
        cache_arrays = dict(tree_flatten(cache_data))
        cache_class_names = [type(cache).__name__ for cache in record_cache]
        safetensor_metadata = dict(
            tree_flatten(
                [
                    cache_meta_states,
                    cache_class_names,
                ]
            )
        )
        snapshot_arrays = {
            name: mx.contiguous(array) for name, array in cache_arrays.items()
        }

        # Schedule snapshot materialization before handing off disk I/O.
        mx.async_eval(list(snapshot_arrays.values()))

        return PreparedPromptRecord(
            key=record_key,
            metadata=PromptCacheRecordMetadata(
                chunk_key=chunk_key,
                record_kind=record_kind,
                layer_indices=layer_indices,
            ),
            snapshot_arrays=snapshot_arrays,
            safetensor_metadata=safetensor_metadata,
        )

    def _touch_cache_entry(self, key: str) -> None:
        total_size = self._blob_store.size(key)
        previous_size = self._key_sizes.get(key, 0)
        self._key_sizes[key] = total_size
        self._total_bytes += total_size - previous_size
        self._lru_keys.pop(key, None)
        self._lru_keys[key] = None

    def _ordered_record_keys_for_touch(
        self,
        chunk_keys: list[str],
        record_keys_by_chunk_key: dict[str, list[str]],
    ) -> list[str]:
        ordered_record_keys = []
        # vLLM/LMCache order the LRU so suffix blocks evict before prefixes.
        for chunk_key in reversed(chunk_keys):
            record_keys = record_keys_by_chunk_key[chunk_key]
            record_keys_by_kind = {
                self._record_metadata_by_key[record_key].record_kind: record_key
                for record_key in record_keys
            }
            # Touch low retention priority first so important records stay newest.
            for record_kind in reversed(_RECORD_RETENTION_PRIORITY):
                record_key = record_keys_by_kind.get(record_kind)
                if record_key is not None:
                    ordered_record_keys.append(record_key)

        return ordered_record_keys

    def _touch_longest_budget_fit_restore_chain(
        self,
        prefix_chunks: list[PromptPrefixChunk],
    ) -> None:
        planner = self._cache_restore_planner()
        for end_idx in range(len(prefix_chunks), 0, -1):
            candidate_chunks = prefix_chunks[:end_idx]
            record_keys_by_chunk_key = planner.restore_record_keys_for_chunk_chain(
                candidate_chunks
            )
            if record_keys_by_chunk_key is None:
                continue

            record_keys = self._ordered_record_keys_for_touch(
                [chunk.key for chunk in candidate_chunks],
                record_keys_by_chunk_key,
            )
            # Preserve the longest complete restore set that can survive eviction.
            if sum(self._key_sizes[key] for key in record_keys) <= (
                self._max_cache_store_bytes
            ):
                for record_key in record_keys:
                    self._touch_cache_entry(record_key)
                return

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self._max_cache_store_bytes:
            key_to_evict = next(iter(self._lru_keys), None)
            if key_to_evict is None:
                break

            self._evict_key(key_to_evict)

    def _maybe_log_cache_usage(self) -> None:
        now = monotonic()
        if (
            self._last_cache_usage_log_time
            and now - self._last_cache_usage_log_time
            < _CACHE_USAGE_LOG_COOLDOWN_SECONDS
        ):
            return

        self._last_cache_usage_log_time = now
        logger.info(
            "VLM prompt cache disk usage: used_mib=%.1f cap_mib=%.1f "
            "lifetime_evicted_mib=%.1f records=%s lifetime=model_load",
            self._total_bytes / _MIB_BYTES,
            self._max_cache_store_bytes / _MIB_BYTES,
            self._cache_evicted_bytes / _MIB_BYTES,
            len(self._record_metadata_by_key),
        )

    def _cache_restore_planner(self) -> PromptCacheRestorePlanner:
        """Return a short-lived read-only view over committed indexes."""
        return PromptCacheRestorePlanner(
            layout=self._require_layout(),
            record_metadata_by_key=self._record_metadata_by_key,
            record_exists=self._blob_store.exists,
        )

    def _require_layout(self) -> PromptCacheLayout:
        if self._layout is None:
            raise RuntimeError("prompt cache layout is not initialized")
        return self._layout

    def _evict_key(self, key: str) -> None:
        evicted_bytes = self._key_sizes.pop(key, 0)
        self._total_bytes -= evicted_bytes
        self._lru_keys.pop(key, None)
        self._record_metadata_by_key.pop(key, None)
        self._cache_evictions += 1
        self._cache_evicted_bytes += evicted_bytes

        self._blob_store.delete(key)
