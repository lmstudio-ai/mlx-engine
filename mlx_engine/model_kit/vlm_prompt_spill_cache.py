from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_engine.model_kit.vlm_prompt_cache_disk_budget import (
    final_spill_cache_cap_bytes,
    provisional_spill_cache_cap_bytes,
)
from mlx_engine.model_kit.vlm_prompt_cache_payload import (
    assemble_prompt_cache_chunks,
    materialize_prompt_state,
    prepare_prompt_cache_payload,
)
from mlx_engine.model_kit.vlm_prompt_cache_index import PromptCacheIndexView
from mlx_engine.model_kit.vlm_prompt_cache_types import (
    PromptCacheChunkMetadata,
    PromptImageSpan,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    PreparedPromptRecord,
    PendingPromptCacheSave,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    RecordKind,
    SpilledPromptState,
    VlmPromptSpillCacheStats,
    build_prefix_cache_chunks,
    make_record_key,
)
from mlx_engine.model_kit.vlm_safetensor_spool import AnonymousSafetensorSpool
from mlx.utils import tree_flatten


_RECORD_TOUCH_ORDER: tuple[RecordKind, ...] = (
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_KV_DELTA,
)


class VlmPromptSpillCache:
    """Actor-owned spill-cache index and anonymous safetensor spool.

    Mutable index/spool operations run on the prompt-cache actor thread. The
    scheduler may prepare immutable save payloads and make advisory cap checks,
    but it must not mutate committed spill state.

    Invariants:
    - A selected restore chain is loaded without interleaved eviction.
    - Selected record keys exist in both metadata and the spool.
    - Touched LRU keys are committed physical records.
    """

    def __init__(self, max_kv_size: int | None = None):
        base_dir = Path("/tmp")
        self._base_dir = base_dir
        self._max_kv_size = max_kv_size
        self._store = AnonymousSafetensorSpool(base_dir)
        self._empirical_cap_set = False
        self._metadata_by_key: dict[str, PromptCacheChunkMetadata] = {}
        self._record_metadata_by_key: dict[str, PromptCacheRecordMetadata] = {}
        self._rope_deltas_by_key: dict[str, Optional[Any]] = {}
        self._key_sizes: dict[str, int] = {}
        self._lru_keys: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._max_cache_bytes = provisional_spill_cache_cap_bytes(base_dir)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def snapshot_stats(self) -> VlmPromptSpillCacheStats:
        """Return best-effort diagnostics for smokes/debug output."""
        total_bytes = self._total_bytes
        max_bytes = self._max_cache_bytes
        entry_count = len(self._record_metadata_by_key)
        hits = self._cache_hits
        misses = self._cache_misses
        evictions = self._cache_evictions
        record_sizes_by_key = dict(self._key_sizes)
        metadata_by_key = dict(self._metadata_by_key)
        record_metadata_by_key = dict(self._record_metadata_by_key)
        chunk_sizes_by_key = {}
        chunk_records_available_by_key = {}
        for chunk_key, metadata in metadata_by_key.items():
            record_keys = [
                make_record_key(chunk_key, record_kind)
                for record_kind in RECORD_WRITE_ORDER
                if record_kind in metadata.payload_kinds
            ]
            chunk_sizes_by_key[chunk_key] = sum(
                record_sizes_by_key.get(record_key, 0) for record_key in record_keys
            )
            chunk_records_available_by_key[chunk_key] = bool(record_keys) and all(
                record_key in record_metadata_by_key
                and record_key in record_sizes_by_key
                for record_key in record_keys
            )

        return VlmPromptSpillCacheStats(
            total_bytes=total_bytes,
            max_bytes=max_bytes,
            entry_count=entry_count,
            hits=hits,
            misses=misses,
            evictions=evictions,
            record_sizes=sorted(record_sizes_by_key.values()),
            record_sizes_by_key=record_sizes_by_key,
            chunk_sizes_by_key=chunk_sizes_by_key,
            chunk_records_available_by_key=chunk_records_available_by_key,
        )

    def restore_longest_prefix(
        self,
        prompt_input_ids: list[int],
        image_spans: list[PromptImageSpan],
    ) -> Optional[SpilledPromptState]:
        """Load the longest matching cacheable prefix.

        The final prompt token stays uncached so generation has a suffix to
        process. Too-short or unchunkable prompts are ineligible and do not
        affect hit/miss stats.
        """
        max_reusable_prefix_len = len(prompt_input_ids) - 1
        if max_reusable_prefix_len <= 0:
            return None

        request_chunks = build_prefix_cache_chunks(
            prompt_input_ids,
            image_spans,
        )
        eligible_chunks = [
            chunk for chunk in request_chunks if chunk.end <= max_reusable_prefix_len
        ]
        if not eligible_chunks:
            return None

        candidate_chunk_keys = []
        for chunk in eligible_chunks:
            # Chunk keys encode prompt tokens plus image identities up to here.
            metadata = self._metadata_by_key.get(chunk.key)

            # Prefix matching must stop at the first missing/divergent chunk.
            if metadata is None or metadata.chunk_end != chunk.end:
                break

            candidate_chunk_keys.append(chunk.key)

        # A later SWA boundary may be restorable even if an earlier boundary is
        # missing old rotating records, so availability selection happens once
        # over the full matching chain.
        restore_records = self._find_best_effort_restore_records(candidate_chunk_keys)
        hit_chunks = 0 if restore_records is None else len(restore_records[1])
        self._record_prefix_lookup(
            hit_chunks=0,
            miss_chunks=len(eligible_chunks) - hit_chunks,
        )
        if restore_records is None:
            return None

        cached_prefix_len, chunk_keys, record_keys_by_chunk_key = restore_records
        return self._load_restore_records(
            cached_prefix_len,
            chunk_keys,
            record_keys_by_chunk_key,
        )

    def _load_restore_records(
        self,
        cached_prefix_len: int,
        chunk_keys: list[str],
        record_keys_by_chunk_key: dict[str, list[str]],
    ) -> SpilledPromptState:
        """Load a selected prompt-cache restore chain."""
        chunk_prompt_caches = []
        chunk_metadata = []
        rope_deltas = None

        # Load each chunk's physical records into sparse per-layer cache lists.
        for key in chunk_keys:
            metadata, prompt_cache, rope_deltas = self._load_one_chunk(
                key,
                record_keys_by_chunk_key[key],
            )
            chunk_metadata.append(metadata)
            chunk_prompt_caches.append(prompt_cache)

        prompt_cache = assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            chunk_metadata,
        )
        # Disk restores run on the prompt-cache actor; decode consumes the cache
        # on the scheduler thread. Force assembled arrays now so no lazy graph
        # keeps a thread-local MLX stream from the restore worker.
        materialize_prompt_state(prompt_cache, rope_deltas)

        # Restore access refreshes exactly the records used by this chain.
        for record_key in self._ordered_record_keys_for_touch(
            chunk_keys,
            record_keys_by_chunk_key,
        ):
            self._touch_cache_entry(record_key)
        # Count chunks only after the restore has actually materialized.
        self._record_prefix_lookup(
            hit_chunks=len(chunk_keys),
            miss_chunks=0,
        )

        return SpilledPromptState(
            cached_prefix_len=cached_prefix_len,
            prompt_cache=prompt_cache,
            rope_deltas=rope_deltas,
        )

    def _load_one_chunk(
        self, key: str, record_keys: list[str]
    ) -> tuple[PromptCacheChunkMetadata, list[Any], Optional[Any]]:
        metadata = self._metadata_by_key[key]
        rope_deltas = self._rope_deltas_by_key.get(key)

        prompt_cache: list[Any] = [None] * len(metadata.payload_kinds)
        for record_key in record_keys:
            record_prompt_cache = self._store.load_prompt_cache(record_key)
            record_metadata = self._record_metadata_by_key[record_key]

            for layer_idx, cache in zip(
                record_metadata.layer_indices, record_prompt_cache
            ):
                prompt_cache[layer_idx] = cache

        return metadata, prompt_cache, rope_deltas

    def _find_best_effort_restore_records(
        self,
        chunk_keys: list[str],
    ) -> Optional[tuple[int, list[str], dict[str, list[str]]]]:
        """Return the longest currently loadable records for an ordered chain."""
        index_view = self._cache_index_view()
        # Loadability is non-monotonic for SWA/state records, so scan downward.
        for end_idx in range(len(chunk_keys), 0, -1):
            candidate_chunk_keys = chunk_keys[:end_idx]
            record_keys_by_chunk_key = index_view.restore_record_keys_for_chunk_chain(
                candidate_chunk_keys
            )
            if record_keys_by_chunk_key is not None:
                cached_prefix_len = self._metadata_by_key[
                    candidate_chunk_keys[-1]
                ].chunk_end
                return cached_prefix_len, candidate_chunk_keys, record_keys_by_chunk_key

        return None

    def can_store_records(self) -> bool:
        """Return False when the spill cache is intentionally hot-only."""
        return self._max_cache_bytes != 0

    def final_cap_from_completed_cache(self, prompt_cache: list[Any]) -> int | None:
        """Return the final empirical cap from a real completed cache."""
        if self._empirical_cap_set:
            return None

        return final_spill_cache_cap_bytes(
            self._base_dir,
            prompt_cache,
            self._max_kv_size,
        )

    def commit_final_cap(self, max_cache_bytes: int) -> None:
        """Set the final cap and evict records from the cache actor thread."""
        if self._empirical_cap_set:
            return
        self._max_cache_bytes = max_cache_bytes
        self._empirical_cap_set = True

        self._evict_if_needed()

    def commit_pending_save(self, pending_save: PendingPromptCacheSave) -> None:
        """Commit a pending save from the cache actor thread."""
        if not self.can_store_records():
            return
        # Chunks may be partial; restore planning checks physical record presence.
        self._metadata_by_key[pending_save.key] = pending_save.metadata
        if pending_save.rope_deltas is None:
            self._rope_deltas_by_key.pop(pending_save.key, None)
        else:
            self._rope_deltas_by_key[pending_save.key] = pending_save.rope_deltas

        try:
            for record in pending_save.records:
                if self._store.exists(record.key):
                    self._record_metadata_by_key[record.key] = record.metadata
                    self._touch_cache_entry(record.key)
                    continue

                # The actor waits, writes, then publishes/account each record.
                mx.eval(list(record.snapshot_arrays.values()))
                self._store.put(
                    record.key,
                    record.snapshot_arrays,
                    record.safetensor_metadata,
                )
                self._record_metadata_by_key[record.key] = record.metadata
                self._touch_cache_entry(record.key)

            chain_record_keys = (
                self._cache_index_view().restore_record_keys_for_chunk_chain(
                    pending_save.metadata.prefix_chunk_keys
                )
            )
            if chain_record_keys is not None:
                for record_key in self._ordered_record_keys_for_touch(
                    pending_save.metadata.prefix_chunk_keys,
                    chain_record_keys,
                ):
                    self._touch_cache_entry(record_key)
        finally:
            self._evict_if_needed()

    def close(self) -> None:
        """Clear metadata and close the actor-owned anonymous spool."""
        self._metadata_by_key.clear()
        self._record_metadata_by_key.clear()
        self._rope_deltas_by_key.clear()
        self._key_sizes.clear()
        self._lru_keys.clear()
        self._total_bytes = 0
        self._store.close()

    def prepare_save(
        self,
        *,
        chunk: PromptPrefixChunk,
        prompt_cache: list[Any],
        rope_deltas: Optional[Any],
    ) -> PendingPromptCacheSave:
        """Prepare a cache save for the background actor."""
        payload_cache, payload_kinds = prepare_prompt_cache_payload(
            prompt_cache,
            chunk.start,
            chunk.end,
        )
        metadata = PromptCacheChunkMetadata(
            chunk_end=chunk.end,
            prefix_chunk_keys=list(chunk.prefix_chunk_keys),
            payload_kinds=payload_kinds,
        )
        records = []
        for record_kind in RECORD_WRITE_ORDER:
            layer_indices = [
                idx
                for idx, payload_kind in enumerate(payload_kinds)
                if payload_kind == record_kind
            ]
            if not layer_indices:
                continue

            records.append(
                self._prepare_record_save(
                    chunk_key=chunk.key,
                    record_kind=record_kind,
                    layer_indices=layer_indices,
                    record_cache=[payload_cache[idx] for idx in layer_indices],
                )
            )

        return PendingPromptCacheSave(
            key=chunk.key,
            metadata=metadata,
            rope_deltas=rope_deltas,
            records=records,
        )

    def _prepare_record_save(
        self,
        *,
        chunk_key: str,
        record_kind: RecordKind,
        layer_indices: list[int],
        record_cache: list[Any],
    ) -> PreparedPromptRecord:
        record_key = make_record_key(chunk_key, record_kind)
        window_size = (
            record_cache[0].max_size
            if record_kind == RECORD_KIND_ROTATING_DELTA
            else None
        )
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

        # Schedule snapshot materialization before handing disk I/O to the actor.
        mx.async_eval(list(snapshot_arrays.values()))

        return PreparedPromptRecord(
            key=record_key,
            metadata=PromptCacheRecordMetadata(
                chunk_key=chunk_key,
                record_kind=record_kind,
                layer_indices=layer_indices,
                window_size=window_size,
            ),
            snapshot_arrays=snapshot_arrays,
            safetensor_metadata=safetensor_metadata,
        )

    def _touch_cache_entry(self, key: str) -> None:
        total_size = self._store.size(key)
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
            # Within one chunk, optional records should age before full KV.
            for record_kind in _RECORD_TOUCH_ORDER:
                record_key = record_keys_by_kind.get(record_kind)
                if record_key is not None:
                    ordered_record_keys.append(record_key)

        return ordered_record_keys

    def _record_prefix_lookup(self, *, hit_chunks: int, miss_chunks: int) -> None:
        self._cache_hits += hit_chunks
        self._cache_misses += miss_chunks

    def _evict_if_needed(self) -> None:
        evicted = False
        while self._total_bytes > self._max_cache_bytes:
            key_to_evict = next(iter(self._lru_keys), None)
            if key_to_evict is None:
                break

            self._evict_key(key_to_evict)
            evicted = True

        if evicted:
            self._prune_unreferenced_chunk_metadata()

    def _cache_index_view(self) -> PromptCacheIndexView:
        """Return a short-lived read-only view over committed indexes."""
        return PromptCacheIndexView(
            metadata_by_key=self._metadata_by_key,
            record_metadata_by_key=self._record_metadata_by_key,
            record_exists=self._store.exists,
        )

    def _evict_key(self, key: str) -> None:
        self._total_bytes -= self._key_sizes.pop(key, 0)
        self._lru_keys.pop(key, None)
        self._record_metadata_by_key.pop(key, None)
        self._cache_evictions += 1

        self._store.delete(key)

    def _prune_unreferenced_chunk_metadata(self) -> None:
        """Drop chunk metadata that no remaining record can restore through."""
        referenced_chunk_keys = set()
        direct_record_chunk_keys = set()
        for record_metadata in self._record_metadata_by_key.values():
            chunk_key = record_metadata.chunk_key
            direct_record_chunk_keys.add(chunk_key)
            chunk_metadata = self._metadata_by_key.get(chunk_key)
            if chunk_metadata is None:
                referenced_chunk_keys.add(chunk_key)
            else:
                referenced_chunk_keys.update(chunk_metadata.prefix_chunk_keys)

        for chunk_key in list(self._metadata_by_key):
            if chunk_key not in referenced_chunk_keys:
                self._metadata_by_key.pop(chunk_key, None)
                self._rope_deltas_by_key.pop(chunk_key, None)

        for chunk_key in list(self._rope_deltas_by_key):
            if chunk_key not in direct_record_chunk_keys:
                self._rope_deltas_by_key.pop(chunk_key, None)
