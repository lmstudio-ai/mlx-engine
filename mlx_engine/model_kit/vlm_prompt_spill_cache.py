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
    PromptCacheLayout,
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


_RECORD_RETENTION_PRIORITY: tuple[RecordKind, ...] = (
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
)


def _make_prompt_cache_layout(
    payload_cache: list[Any],
    payload_kinds: list[RecordKind],
) -> PromptCacheLayout:
    layer_indices_by_kind: dict[RecordKind, list[int]] = {}
    rotating_window_size = None
    for layer_idx, record_kind in enumerate(payload_kinds):
        layer_indices_by_kind.setdefault(record_kind, []).append(layer_idx)
        if record_kind == RECORD_KIND_ROTATING_DELTA:
            window_size = int(payload_cache[layer_idx].max_size)
            rotating_window_size = max(rotating_window_size or 0, window_size)

    return PromptCacheLayout(
        layer_kinds=list(payload_kinds),
        layer_indices_by_kind=layer_indices_by_kind,
        rotating_window_size=rotating_window_size,
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
        self._layout: PromptCacheLayout | None = None
        self._record_metadata_by_key: dict[str, PromptCacheRecordMetadata] = {}
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
                record_key in record_sizes_by_key and self._store.exists(record_key)
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
        if self._layout is None:
            self._record_prefix_lookup(
                hit_chunks=0,
                miss_chunks=len(eligible_chunks),
            )
            return None

        # A later SWA boundary may be restorable even if an earlier boundary is
        # missing old rotating records, so availability selection happens once
        # over the full matching chain.
        restore_records = self._find_longest_loadable_records(eligible_chunks)
        hit_chunks = 0 if restore_records is None else len(restore_records[1])
        self._record_prefix_lookup(
            hit_chunks=0,
            miss_chunks=len(eligible_chunks) - hit_chunks,
        )
        if restore_records is None:
            return None

        cached_prefix_len, chunks, record_keys_by_chunk_key = restore_records
        return self._load_restore_records(
            cached_prefix_len,
            chunks,
            record_keys_by_chunk_key,
        )

    def _load_restore_records(
        self,
        cached_prefix_len: int,
        chunks: list[PromptPrefixChunk],
        record_keys_by_chunk_key: dict[str, list[str]],
    ) -> SpilledPromptState:
        """Load a selected prompt-cache restore chain."""
        layout = self._layout
        assert layout is not None

        chunk_prompt_caches = []

        # Load each chunk's physical records into sparse per-layer cache lists.
        for chunk in chunks:
            prompt_cache = self._load_one_chunk(
                record_keys_by_chunk_key[chunk.key],
                layout,
            )
            chunk_prompt_caches.append(prompt_cache)

        prompt_cache = assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            chunks,
            layout,
        )
        # Disk restores run on the prompt-cache actor; decode consumes the cache
        # on the scheduler thread. Force assembled arrays now so no lazy graph
        # keeps a thread-local MLX stream from the restore worker.
        materialize_prompt_state(prompt_cache)

        # Restore access refreshes exactly the records used by this chain.
        for record_key in self._ordered_record_keys_for_touch(
            [chunk.key for chunk in chunks],
            record_keys_by_chunk_key,
        ):
            self._touch_cache_entry(record_key)
        # Count chunks only after the restore has actually materialized.
        self._record_prefix_lookup(
            hit_chunks=len(chunks),
            miss_chunks=0,
        )

        return SpilledPromptState(
            cached_prefix_len=cached_prefix_len,
            prompt_cache=prompt_cache,
        )

    def _load_one_chunk(
        self,
        record_keys: list[str],
        layout: PromptCacheLayout,
    ) -> list[Any]:
        prompt_cache: list[Any] = [None] * len(layout.layer_kinds)
        for record_key in record_keys:
            record_prompt_cache = self._store.load_prompt_cache(record_key)
            record_metadata = self._record_metadata_by_key[record_key]

            for layer_idx, cache in zip(
                record_metadata.layer_indices, record_prompt_cache
            ):
                prompt_cache[layer_idx] = cache

        return prompt_cache

    def _find_longest_loadable_records(
        self,
        chunks: list[PromptPrefixChunk],
    ) -> Optional[tuple[int, list[PromptPrefixChunk], dict[str, list[str]]]]:
        """Return the longest currently loadable records for an ordered chain."""
        index_view = self._cache_index_view()
        # Loadability is non-monotonic for SWA/state records, so scan downward.
        for end_idx in range(len(chunks), 0, -1):
            candidate_chunks = chunks[:end_idx]
            record_keys_by_chunk_key = index_view.restore_record_keys_for_chunk_chain(
                candidate_chunks
            )
            if record_keys_by_chunk_key is not None:
                cached_prefix_len = candidate_chunks[-1].end
                return cached_prefix_len, candidate_chunks, record_keys_by_chunk_key

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
        if self._layout is None:
            self._layout = pending_save.cache_layout

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
                    pending_save.prefix_chunks
                )
            )
            if chain_record_keys is not None:
                for record_key in self._ordered_record_keys_for_touch(
                    [chunk.key for chunk in pending_save.prefix_chunks],
                    chain_record_keys,
                ):
                    self._touch_cache_entry(record_key)
        finally:
            self._evict_if_needed()

    def close(self) -> None:
        """Clear metadata and close the actor-owned anonymous spool."""
        self._layout = None
        self._record_metadata_by_key.clear()
        self._key_sizes.clear()
        self._lru_keys.clear()
        self._total_bytes = 0
        self._store.close()

    def prepare_save(
        self,
        *,
        chunk: PromptPrefixChunk,
        prefix_chunks: list[PromptPrefixChunk],
        prompt_cache: list[Any],
    ) -> PendingPromptCacheSave:
        """Prepare a cache save for the background actor."""
        payload_cache, payload_kinds = prepare_prompt_cache_payload(
            prompt_cache,
            chunk.start,
            chunk.end,
        )
        layout = _make_prompt_cache_layout(payload_cache, payload_kinds)
        records = []
        for record_kind in RECORD_WRITE_ORDER:
            layer_indices = layout.layer_indices_by_kind.get(record_kind, [])
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
            prefix_chunks=prefix_chunks,
            cache_layout=layout,
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
            # Touch low retention priority first so important records stay newest.
            for record_kind in reversed(_RECORD_RETENTION_PRIORITY):
                record_key = record_keys_by_kind.get(record_kind)
                if record_key is not None:
                    ordered_record_keys.append(record_key)

        return ordered_record_keys

    def _record_prefix_lookup(self, *, hit_chunks: int, miss_chunks: int) -> None:
        self._cache_hits += hit_chunks
        self._cache_misses += miss_chunks

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self._max_cache_bytes:
            key_to_evict = next(iter(self._lru_keys), None)
            if key_to_evict is None:
                break

            self._evict_key(key_to_evict)

    def _cache_index_view(self) -> PromptCacheIndexView:
        """Return a short-lived read-only view over committed indexes."""
        layout = self._layout
        assert layout is not None
        return PromptCacheIndexView(
            layout=layout,
            record_metadata_by_key=self._record_metadata_by_key,
            record_exists=self._store.exists,
        )

    def _evict_key(self, key: str) -> None:
        self._total_bytes -= self._key_sizes.pop(key, 0)
        self._lru_keys.pop(key, None)
        self._record_metadata_by_key.pop(key, None)
        self._cache_evictions += 1

        self._store.delete(key)
