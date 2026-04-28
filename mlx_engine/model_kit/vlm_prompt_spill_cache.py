from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Any, Optional

import mlx.core as mx
from mlx_engine.model_kit.vlm_prompt_cache_payload import (
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    assemble_prompt_cache_chunks,
    deserialize_rope_deltas,
    materialize_prompt_state,
    prepare_prompt_cache_payload,
    serialize_rope_deltas,
)
from mlx_engine.model_kit.vlm_prompt_cache_planner import PromptCacheDependencyPlanner
from mlx_engine.model_kit.vlm_prompt_cache_types import (
    CachedPrefixMatch,
    CachedPromptMetadata,
    CachedPromptRecordMetadata,
    PrefixCacheChunk,
    PreparedPromptRecord,
    PreparedPromptSnapshot,
    SpilledPromptState,
    VlmPromptSpillCacheStats,
    build_prefix_cache_chunks,
    make_record_key,
)
from mlx_engine.model_kit.vlm_safetensor_spool import TempSafetensorSpool
from mlx.utils import tree_flatten


MLX_VLM_BATCHED_VISION_DISK_CACHE_DIR_ENV_VAR = (
    "MLX_ENGINE_MLX_VLM_BATCHED_VISION_DISK_CACHE_DIR"
)
DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_DIR = Path(
    "/tmp/mlx-engine-vlm-batched-vision-cache"
)
MLX_VLM_BATCHED_VISION_DISK_CACHE_MAX_BYTES_ENV_VAR = (
    "MLX_ENGINE_MLX_VLM_BATCHED_VISION_DISK_CACHE_MAX_BYTES"
)
DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_FALLBACK_MAX_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_HARD_CEILING_BYTES = 64 * 1024 * 1024 * 1024
DEFAULT_KV_CACHE_DTYPE_BYTES = 2


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_int(config: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    value = _config_value(config, key)
    if value is None:
        return default
    return int(value)


def _text_config(model_config: Any) -> Any:
    return _config_value(model_config, "text_config", model_config)


def _kv_cache_dtype_bytes(config: Any) -> int:
    dtype = str(
        _config_value(config, "torch_dtype") or _config_value(config, "dtype") or ""
    ).lower()
    if "64" in dtype:
        return 8
    if "32" in dtype:
        return 4
    return DEFAULT_KV_CACHE_DTYPE_BYTES


def _estimate_model_kv_cache_max_bytes(model_config: Any) -> Optional[int]:
    if model_config is None:
        return None

    config = _text_config(model_config)
    max_positions = _config_int(config, "max_position_embeddings")
    hidden_size = _config_int(config, "hidden_size")
    num_attention_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads", num_attention_heads)
    head_dim = _config_int(config, "head_dim")
    if head_dim is None and hidden_size is not None and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    num_layers = _config_int(config, "num_hidden_layers") or _config_int(
        config, "num_layers"
    )
    layer_types = list(_config_value(config, "layer_types", []) or [])
    if num_layers is None:
        num_layers = len(layer_types)
    if not layer_types and num_layers is not None:
        layer_types = ["full_attention"] * num_layers

    shared_kv_layers = _config_int(config, "num_kv_shared_layers", 0) or 0
    cache_layer_count = max(0, (num_layers or len(layer_types)) - shared_kv_layers)
    layer_types = layer_types[:cache_layer_count]

    if not max_positions or not num_kv_heads or not head_dim or not layer_types:
        return None

    dtype_bytes = _kv_cache_dtype_bytes(config)
    total_bytes = 0
    for layer_type in layer_types:
        if layer_type == "linear_attention":
            continue

        layer_seq_len = max_positions
        layer_head_dim = head_dim
        layer_kv_heads = num_kv_heads
        if layer_type == "sliding_attention":
            layer_seq_len = _config_int(config, "sliding_window", max_positions)
        elif layer_type == "full_attention":
            layer_head_dim = _config_int(config, "global_head_dim", head_dim)
            layer_kv_heads = _config_int(
                config, "num_global_key_value_heads", num_kv_heads
            )

        total_bytes += 2 * layer_seq_len * layer_kv_heads * layer_head_dim * dtype_bytes

    return total_bytes or None


def _default_max_cache_bytes(cache_dir: Path, model_config: Any) -> int:
    half_free_disk = shutil.disk_usage(cache_dir).free // 2
    model_kv_bytes = (
        _estimate_model_kv_cache_max_bytes(model_config)
        or DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_FALLBACK_MAX_BYTES
    )
    return min(
        half_free_disk,
        model_kv_bytes,
        DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_HARD_CEILING_BYTES,
    )


class VlmPromptSpillCache:
    def __init__(self, model_config: Any = None):
        base_dir = Path(
            os.environ.get(
                MLX_VLM_BATCHED_VISION_DISK_CACHE_DIR_ENV_VAR,
                DEFAULT_MLX_VLM_BATCHED_VISION_DISK_CACHE_DIR,
            )
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        self._store = TempSafetensorSpool(base_dir)
        self._metadata_index_lock = Lock()
        self._metadata_by_key: dict[str, CachedPromptMetadata] = {}
        self._record_metadata_by_key: dict[str, CachedPromptRecordMetadata] = {}
        self._dependency_planner = PromptCacheDependencyPlanner(
            metadata_by_key=self._metadata_by_key,
            record_metadata_by_key=self._record_metadata_by_key,
            record_exists=self._store.exists,
        )
        self._serialized_rope_deltas_by_key: dict[str, Optional[Any]] = {}
        self._key_sizes: dict[str, int] = {}
        self._lru_keys: OrderedDict[str, None] = OrderedDict()
        self._total_bytes = 0
        self._max_cache_bytes = self._read_max_cache_bytes(base_dir, model_config)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._pending_save_keys: set[str] = set()
        self._pending_save_keys_lock = Lock()

    def snapshot_stats(self) -> VlmPromptSpillCacheStats:
        with self._metadata_index_lock:
            total_bytes = self._total_bytes
            max_bytes = self._max_cache_bytes
            entry_count = len(self._record_metadata_by_key)
            hits = self._cache_hits
            misses = self._cache_misses
            evictions = self._cache_evictions
        with self._pending_save_keys_lock:
            pending_saves = len(self._pending_save_keys)

        return VlmPromptSpillCacheStats(
            total_bytes=total_bytes,
            max_bytes=max_bytes,
            entry_count=entry_count,
            pending_saves=pending_saves,
            hits=hits,
            misses=misses,
            evictions=evictions,
        )

    def load_chunk_sequence(self, keys: list[str]) -> Optional[SpilledPromptState]:
        record_keys_by_chunk_key = self._record_keys_for_chunk_sequence(keys)
        if record_keys_by_chunk_key is None:
            return None

        chunk_prompt_caches = []
        chunk_metadata = []
        rope_deltas = None

        for key in keys:
            loaded = self._load_one_chunk(key, record_keys_by_chunk_key[key])
            if loaded is None:
                return None
            metadata, prompt_cache, rope_deltas = loaded
            chunk_metadata.append(metadata)
            chunk_prompt_caches.append(prompt_cache)

        if not chunk_prompt_caches:
            return None

        prompt_cache = assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            chunk_metadata,
        )
        # Restore can happen on a background thread; decode consumes the cache on
        # the scheduler thread. Force assembled arrays now so no lazy graph keeps
        # a thread-local MLX stream from the restore worker.
        materialize_prompt_state(prompt_cache, rope_deltas)

        return SpilledPromptState(
            prompt_cache=prompt_cache,
            rope_deltas=rope_deltas,
        )

    def _load_one_chunk(
        self, key: str, record_keys: list[str]
    ) -> Optional[tuple[CachedPromptMetadata, list[Any], Optional[Any]]]:
        if self._is_save_pending(key):
            return None

        with self._metadata_index_lock:
            metadata = self._metadata_by_key.get(key)
            serialized_rope_deltas = self._serialized_rope_deltas_by_key.get(key)
        if metadata is None:
            return None

        prompt_cache: list[Any] = [None] * len(metadata.payload_kinds)
        for record_key in record_keys:
            if not self._store.exists(record_key):
                return None

            record_prompt_cache = self._store.load_prompt_cache(record_key)
            with self._metadata_index_lock:
                record_metadata = self._record_metadata_by_key.get(record_key)
            if record_metadata is None:
                return None

            for layer_idx, cache in zip(
                record_metadata.layer_indices, record_prompt_cache
            ):
                prompt_cache[layer_idx] = cache

            self._touch_cache_entry(record_key)

        if any(
            cache is None
            and metadata.payload_kinds[idx]
            not in {RECORD_KIND_ROTATING_DELTA, RECORD_KIND_STATE_CHECKPOINT}
            for idx, cache in enumerate(prompt_cache)
        ):
            return None

        rope_deltas = deserialize_rope_deltas(serialized_rope_deltas)

        return metadata, prompt_cache, rope_deltas

    def find_longest_prefix(
        self,
        prompt_input_ids: list[int],
        image_hashes: list[str],
        min_reusable_prefix_len: int,
    ) -> Optional[CachedPrefixMatch]:
        max_reusable_prefix_len = len(prompt_input_ids) - 1
        if max_reusable_prefix_len <= 0:
            self._record_prefix_lookup(hit=False)
            return None

        request_chunks = build_prefix_cache_chunks(
            prompt_input_ids,
            image_hashes,
            min_reusable_prefix_len,
        )
        if not request_chunks:
            self._record_prefix_lookup(hit=False)
            return None

        longest_match = None
        candidate_chunk_keys = []
        for chunk in request_chunks:
            if chunk.end > max_reusable_prefix_len:
                break

            if self._is_save_pending(chunk.key):
                break

            with self._metadata_index_lock:
                metadata = self._metadata_by_key.get(chunk.key)

            if (
                metadata is None
                or metadata.image_hashes != list(image_hashes)
                or metadata.chunk_end != chunk.end
                or metadata.chunk_hash != chunk.chunk_hash
            ):
                break

            candidate_chunk_keys.append(chunk.key)
            # An early boundary can be unloadable after old SWA records are
            # evicted, while a later boundary only needs the final SWA window.
            with self._metadata_index_lock:
                records_available = (
                    self._record_keys_for_chunk_sequence_locked(candidate_chunk_keys)
                    is not None
                )
            if records_available:
                longest_match = CachedPrefixMatch(
                    key=chunk.key,
                    metadata=metadata,
                    matched_prefix_len=chunk.end,
                    chunk_keys=list(candidate_chunk_keys),
                )

        self._record_prefix_lookup(hit=longest_match is not None)
        return longest_match

    def prepare_save(
        self,
        *,
        chunk: PrefixCacheChunk,
        prompt_input_ids: list[int],
        image_hashes: list[str],
        min_reusable_prefix_len: int,
        prompt_cache: list[Any],
        rope_deltas: Optional[Any],
    ) -> Optional[PreparedPromptSnapshot]:
        key = chunk.key
        if not self._begin_save(key):
            return

        try:
            return self._prepare_save_now(
                chunk=chunk,
                prompt_input_ids=prompt_input_ids,
                image_hashes=image_hashes,
                min_reusable_prefix_len=min_reusable_prefix_len,
                prompt_cache=prompt_cache,
                rope_deltas=rope_deltas,
            )
        except Exception:
            self._finish_save(key)
            raise

    def commit_prepared_save(self, prepared_snapshot: PreparedPromptSnapshot) -> None:
        try:
            self._write_prepared_save(prepared_snapshot)
            self._index_metadata(prepared_snapshot.key, prepared_snapshot.metadata)
            self._index_record_metadata(prepared_snapshot.records)
            self._set_serialized_rope_deltas(
                prepared_snapshot.key, prepared_snapshot.serialized_rope_deltas
            )
            for record in prepared_snapshot.records:
                self._touch_cache_entry(record.key)
        finally:
            self._finish_save(prepared_snapshot.key)
            self._evict_if_needed()

    def discard_prepared_save(self, prepared_snapshot: PreparedPromptSnapshot) -> None:
        self._finish_save(prepared_snapshot.key)

    def close(self) -> None:
        with self._metadata_index_lock:
            self._metadata_by_key.clear()
            self._record_metadata_by_key.clear()
            self._serialized_rope_deltas_by_key.clear()
            self._key_sizes.clear()
            self._lru_keys.clear()
            self._total_bytes = 0
        self._store.close()

    def _begin_save(self, key: str) -> bool:
        with self._pending_save_keys_lock:
            if key in self._pending_save_keys:
                return False
            self._pending_save_keys.add(key)
            return True

    def _finish_save(self, key: str) -> None:
        with self._pending_save_keys_lock:
            self._pending_save_keys.discard(key)

    def _is_save_pending(self, key: str) -> bool:
        with self._pending_save_keys_lock:
            return key in self._pending_save_keys

    def _record_keys_for_chunk_sequence(
        self, chunk_keys: list[str]
    ) -> Optional[dict[str, list[str]]]:
        if not chunk_keys:
            return {}

        with self._metadata_index_lock:
            return self._record_keys_for_chunk_sequence_locked(chunk_keys)

    def _record_keys_for_chunk_sequence_locked(
        self, chunk_keys: list[str]
    ) -> Optional[dict[str, list[str]]]:
        return self._dependency_planner.record_keys_for_chunk_sequence(chunk_keys)

    def _prepare_save_now(
        self,
        *,
        chunk: PrefixCacheChunk,
        prompt_input_ids: list[int],
        image_hashes: list[str],
        min_reusable_prefix_len: int,
        prompt_cache: list[Any],
        rope_deltas: Optional[Any],
    ) -> PreparedPromptSnapshot:
        payload_cache, payload_kinds = prepare_prompt_cache_payload(
            prompt_cache,
            chunk.start,
            chunk.end,
        )
        metadata = CachedPromptMetadata(
            prompt_input_ids=prompt_input_ids,
            image_hashes=image_hashes,
            min_reusable_prefix_len=min_reusable_prefix_len,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            chunk_hash=chunk.chunk_hash,
            payload_kinds=payload_kinds,
        )
        serialized_rope_deltas = serialize_rope_deltas(rope_deltas)

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
                    prompt_input_ids=prompt_input_ids,
                )
            )

        return PreparedPromptSnapshot(
            key=chunk.key,
            metadata=metadata,
            serialized_rope_deltas=serialized_rope_deltas,
            records=records,
        )

    def _prepare_record_save(
        self,
        *,
        chunk_key: str,
        record_kind: str,
        layer_indices: list[int],
        record_cache: list[Any],
        prompt_input_ids: list[int],
    ) -> PreparedPromptRecord:
        record_key = make_record_key(chunk_key, record_kind)
        window_size = (
            record_cache[0].max_size
            if record_kind == RECORD_KIND_ROTATING_DELTA
            else None
        )
        cache_data = [cache.state for cache in record_cache]
        cache_info = [cache.meta_state for cache in record_cache]
        cache_arrays = dict(tree_flatten(cache_data))
        cache_classes = [type(cache).__name__ for cache in record_cache]
        snapshot_metadata = dict(
            tree_flatten(
                [
                    cache_info,
                    {
                        "prompt_input_ids": json.dumps(prompt_input_ids),
                        "vlm_record_kind": record_kind,
                        "vlm_layer_indices": json.dumps(layer_indices),
                    },
                    cache_classes,
                ]
            )
        )
        snapshot_arrays = {
            name: mx.contiguous(array) for name, array in cache_arrays.items()
        }

        # Keep MLX array normalization and scheduling on the scheduler thread.
        mx.async_eval(list(snapshot_arrays.values()))

        return PreparedPromptRecord(
            key=record_key,
            metadata=CachedPromptRecordMetadata(
                chunk_key=chunk_key,
                record_kind=record_kind,
                layer_indices=layer_indices,
                window_size=window_size,
            ),
            snapshot_arrays=snapshot_arrays,
            snapshot_metadata=snapshot_metadata,
        )

    def _write_prepared_save(self, prepared_snapshot: PreparedPromptSnapshot) -> None:
        saved_record_keys = []

        try:
            # A future writer thread can block here after the scheduler thread
            # has already kicked off async_eval during prepare_save().
            for record in prepared_snapshot.records:
                mx.eval(list(record.snapshot_arrays.values()))
                self._store.put(
                    record.key,
                    record.snapshot_arrays,
                    record.snapshot_metadata,
                )
                saved_record_keys.append(record.key)
        except Exception:
            for record_key in saved_record_keys:
                self._store.delete(record_key)
            raise

    def _index_metadata(self, key: str, metadata: CachedPromptMetadata) -> None:
        with self._metadata_index_lock:
            self._metadata_by_key[key] = metadata

    def _index_record_metadata(self, records: list[PreparedPromptRecord]) -> None:
        with self._metadata_index_lock:
            for record in records:
                self._record_metadata_by_key[record.key] = record.metadata

    def _remove_record_metadata(self, key: str) -> None:
        with self._metadata_index_lock:
            self._record_metadata_by_key.pop(key, None)

    def _set_serialized_rope_deltas(
        self, key: str, serialized_rope_deltas: Optional[Any]
    ) -> None:
        with self._metadata_index_lock:
            if serialized_rope_deltas is None:
                self._serialized_rope_deltas_by_key.pop(key, None)
            else:
                self._serialized_rope_deltas_by_key[key] = serialized_rope_deltas

    def _read_max_cache_bytes(self, base_dir: Path, model_config: Any) -> Optional[int]:
        raw = os.environ.get(MLX_VLM_BATCHED_VISION_DISK_CACHE_MAX_BYTES_ENV_VAR, "")
        if raw == "":
            return _default_max_cache_bytes(base_dir, model_config)

        max_cache_bytes = int(raw)
        return None if max_cache_bytes <= 0 else max_cache_bytes

    def _touch_cache_entry(self, key: str) -> None:
        total_size = self._get_cache_entry_size(key)
        if total_size == 0:
            return

        with self._metadata_index_lock:
            previous_size = self._key_sizes.get(key, 0)
            self._key_sizes[key] = total_size
            self._total_bytes += total_size - previous_size
            self._lru_keys.pop(key, None)
            self._lru_keys[key] = None

    def _record_prefix_lookup(self, *, hit: bool) -> None:
        with self._metadata_index_lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def _get_cache_entry_size(self, key: str) -> int:
        return self._store.size(key)

    def _get_chunk_record_keys(self, chunk_key: str) -> list[str]:
        with self._metadata_index_lock:
            metadata = self._metadata_by_key.get(chunk_key)
            if metadata is None:
                return []
            return self._get_chunk_record_keys_locked(chunk_key, metadata)

    def _get_chunk_record_keys_locked(
        self, chunk_key: str, metadata: CachedPromptMetadata
    ) -> list[str]:
        return self._dependency_planner.chunk_record_keys(chunk_key, metadata)

    def _get_chunk_cache_entry_size(self, chunk_key: str) -> int:
        return sum(
            self._get_cache_entry_size(record_key)
            for record_key in self._get_chunk_record_keys(chunk_key)
        )

    def _chunk_records_exist(self, chunk_key: str) -> bool:
        with self._metadata_index_lock:
            return self._chunk_records_available_locked(chunk_key)

    def _chunk_records_available_locked(self, chunk_key: str) -> bool:
        metadata = self._metadata_by_key.get(chunk_key)
        if metadata is None:
            return False

        record_keys = self._get_chunk_record_keys_locked(chunk_key, metadata)
        if not record_keys:
            return False

        return all(
            record_key in self._record_metadata_by_key
            and self._store.exists(record_key)
            for record_key in record_keys
        )

    def _chunk_has_live_records_locked(self, chunk_key: str) -> bool:
        return self._dependency_planner.chunk_has_live_records(chunk_key)

    def _debug_record_sizes(self) -> list[int]:
        return self._store.record_sizes()

    def _evict_if_needed(self) -> None:
        if self._max_cache_bytes is None:
            return

        while True:
            with self._metadata_index_lock:
                if self._total_bytes <= self._max_cache_bytes:
                    return

                key_to_evict = self._select_eviction_key_locked()
            if key_to_evict is None:
                return

            self._evict_key(key_to_evict)

    def _select_eviction_key_locked(self) -> Optional[str]:
        stale_optional_key = self._select_stale_optional_eviction_key_locked()
        if stale_optional_key is not None:
            return stale_optional_key

        for key in self._lru_keys:
            if self._is_save_pending(key):
                continue

            record_metadata = self._record_metadata_by_key.get(key)
            if record_metadata is None:
                return key

            chunk_metadata = self._metadata_by_key.get(record_metadata.chunk_key)
            if (
                chunk_metadata is not None
                and self._dependency_planner.has_dependent_chunks(
                    record_metadata.chunk_key, chunk_metadata
                )
            ):
                continue

            return key

        return None

    def _select_stale_optional_eviction_key_locked(self) -> Optional[str]:
        for key in self._lru_keys:
            if self._is_save_pending(key):
                continue

            record_metadata = self._record_metadata_by_key.get(key)
            if record_metadata is None or record_metadata.record_kind not in {
                RECORD_KIND_ROTATING_DELTA,
                RECORD_KIND_STATE_CHECKPOINT,
            }:
                continue

            chunk_metadata = self._metadata_by_key.get(record_metadata.chunk_key)
            if chunk_metadata is None:
                continue
            if self._dependency_planner.is_stale_optional_record(
                record_metadata, chunk_metadata
            ):
                return key

        return None

    def _evict_key(self, key: str) -> None:
        self._store.delete(key)

        with self._metadata_index_lock:
            self._total_bytes -= self._key_sizes.pop(key, 0)
            self._lru_keys.pop(key, None)
            self._cache_evictions += 1

        self._remove_record_metadata(key)
