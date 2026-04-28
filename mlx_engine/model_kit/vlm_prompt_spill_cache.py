from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import mlx.core as mx
from mlx_engine.model_kit.vlm_safetensor_spool import TempSafetensorSpool
from mlx_lm.models.cache import KVCache, RotatingKVCache
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
# LMCache defaults to 256-token external chunks, and MLX KV caches allocate in
# 256-token steps. This is a spill-cache chunk size, not vLLM's KV page size.
DEFAULT_PREFIX_CHUNK_SIZE = 256
DEFAULT_KV_CACHE_DTYPE_BYTES = 2
RECORD_KIND_KV_DELTA = "kv_delta"
RECORD_KIND_ROTATING_DELTA = "rotating_delta"
RECORD_KIND_STATE_CHECKPOINT = "state_checkpoint"
RECORD_WRITE_ORDER = (
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
)
PAYLOAD_KIND_BOUNDARY = RECORD_KIND_STATE_CHECKPOINT
PAYLOAD_KIND_KV_DELTA = RECORD_KIND_KV_DELTA


@dataclass
class SpilledPromptState:
    prompt_cache: list[Any]
    rope_deltas: Optional[Any]


@dataclass
class PrefixCacheChunk:
    start: int
    end: int
    key: str
    chunk_hash: str
    image_hashes: list[str]


@dataclass
class CachedPromptMetadata:
    prompt_input_ids: list[int]
    image_hashes: list[str]
    min_reusable_prefix_len: int
    chunk_start: int
    chunk_end: int
    chunk_hash: str
    payload_kinds: list[str]


@dataclass
class CachedPromptRecordMetadata:
    chunk_key: str
    record_kind: str
    layer_indices: list[int]
    window_size: Optional[int] = None


@dataclass
class CachedPrefixMatch:
    key: str
    metadata: CachedPromptMetadata
    matched_prefix_len: int
    chunk_keys: list[str]


@dataclass
class PreparedPromptRecord:
    key: str
    metadata: CachedPromptRecordMetadata
    snapshot_arrays: dict[str, Any]
    snapshot_metadata: dict[str, str]


@dataclass
class PreparedPromptSnapshot:
    key: str
    metadata: CachedPromptMetadata
    serialized_rope_deltas: Optional[Any]
    records: list[PreparedPromptRecord]


@dataclass
class VlmPromptSpillCacheStats:
    total_bytes: int
    max_bytes: Optional[int]
    entry_count: int
    pending_saves: int
    hits: int
    misses: int
    evictions: int


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


def _serialize_rope_deltas(rope_deltas: Optional[Any]) -> Optional[Any]:
    if rope_deltas is None:
        return None

    # Keep the spill payload on disk; tiny RoPE state can stay process-local.
    return mx.array(rope_deltas, dtype=mx.int32).tolist()


def _deserialize_rope_deltas(
    serialized_rope_deltas: Optional[Any],
) -> Optional[Any]:
    if serialized_rope_deltas is None:
        return None

    return mx.array(serialized_rope_deltas, dtype=mx.int32)


def _is_kv_cache(cache: Any) -> bool:
    # mlx-vlm re-exports mlx-lm cache classes; keep this name-based so local
    # forks do not need identical module identities.
    return type(cache).__name__ == "KVCache"


def _is_rotating_kv_cache(cache: Any) -> bool:
    return type(cache).__name__ == "RotatingKVCache" and getattr(cache, "keep", 0) == 0


def _slice_kv_cache(cache: Any, chunk_start: int, chunk_end: int) -> KVCache:
    keys, values = cache.state
    chunk_cache = KVCache()
    chunk_cache.state = (
        mx.contiguous(keys[..., chunk_start:chunk_end, :]),
        mx.contiguous(values[..., chunk_start:chunk_end, :]),
    )
    return chunk_cache


def _slice_rotating_kv_cache(
    cache: Any,
    chunk_start: int,
    chunk_end: int,
) -> RotatingKVCache:
    keys, values = cache.state
    window_start = cache.offset - keys.shape[2]
    local_start = max(0, chunk_start - window_start)
    local_end = min(keys.shape[2], chunk_end - window_start)

    chunk_cache = RotatingKVCache(max_size=cache.max_size, keep=cache.keep)
    chunk_cache.state = (
        mx.contiguous(keys[..., local_start:local_end, :]),
        mx.contiguous(values[..., local_start:local_end, :]),
    )
    chunk_cache.offset = chunk_end
    chunk_cache._idx = local_end - local_start
    return chunk_cache


def _prepare_prompt_cache_payload(
    prompt_cache: list[Any],
    chunk_start: int,
    chunk_end: int,
) -> tuple[list[Any], list[str]]:
    payload_cache = []
    payload_kinds = []
    for cache in prompt_cache:
        if _is_kv_cache(cache):
            payload_cache.append(_slice_kv_cache(cache, chunk_start, chunk_end))
            payload_kinds.append(PAYLOAD_KIND_KV_DELTA)
        elif _is_rotating_kv_cache(cache):
            payload_cache.append(
                _slice_rotating_kv_cache(cache, chunk_start, chunk_end)
            )
            payload_kinds.append(RECORD_KIND_ROTATING_DELTA)
        else:
            # Opaque array-state caches stay as exact boundary checkpoints for V1.
            payload_cache.append(cache)
            payload_kinds.append(PAYLOAD_KIND_BOUNDARY)

    return payload_cache, payload_kinds


def _concat_kv_delta_caches(caches: list[Any]) -> KVCache:
    keys = mx.concatenate([cache.state[0] for cache in caches], axis=2)
    values = mx.concatenate([cache.state[1] for cache in caches], axis=2)
    cache = KVCache()
    cache.state = (mx.contiguous(keys), mx.contiguous(values))
    return cache


def _concat_rotating_delta_caches(
    caches: list[Any],
    final_offset: int,
) -> RotatingKVCache:
    keys = mx.concatenate([cache.state[0] for cache in caches], axis=2)
    values = mx.concatenate([cache.state[1] for cache in caches], axis=2)
    max_size = caches[-1].max_size
    keep = caches[-1].keep
    if keys.shape[2] > max_size:
        keys = keys[..., -max_size:, :]
        values = values[..., -max_size:, :]

    cache = RotatingKVCache(max_size=max_size, keep=keep)
    cache.state = (mx.contiguous(keys), mx.contiguous(values))
    cache.offset = final_offset
    cache._idx = keys.shape[2]
    return cache


def _assemble_prompt_cache_chunks(
    chunk_prompt_caches: list[list[Any]],
    chunk_metadata: list["CachedPromptMetadata"],
) -> list[Any]:
    if len(chunk_prompt_caches) == 1:
        return chunk_prompt_caches[0]

    assembled = []
    layer_count = len(chunk_prompt_caches[-1])
    for layer_idx in range(layer_count):
        layer_kinds = [metadata.payload_kinds[layer_idx] for metadata in chunk_metadata]
        layer_chunks = [prompt_cache[layer_idx] for prompt_cache in chunk_prompt_caches]
        if all(kind == PAYLOAD_KIND_KV_DELTA for kind in layer_kinds):
            assembled.append(_concat_kv_delta_caches(layer_chunks))
        elif all(kind == RECORD_KIND_ROTATING_DELTA for kind in layer_kinds):
            assembled.append(
                _concat_rotating_delta_caches(
                    [cache for cache in layer_chunks if cache is not None],
                    chunk_metadata[-1].chunk_end,
                )
            )
        else:
            assembled.append(
                next(cache for cache in reversed(layer_chunks) if cache is not None)
            )

    return assembled


def _materialize_prompt_state(
    prompt_cache: list[Any],
    rope_deltas: Optional[Any],
) -> None:
    eval_targets = tree_flatten([cache.state for cache in prompt_cache])
    if rope_deltas is not None:
        eval_targets.append(("rope_deltas", rope_deltas))
    if eval_targets:
        mx.eval([value for _, value in eval_targets])


def _build_chunk_metadata(
    prompt_input_ids: list[int],
    image_hashes: list[str],
    min_reusable_prefix_len: int,
    chunk_size: int = DEFAULT_PREFIX_CHUNK_SIZE,
) -> list[PrefixCacheChunk]:
    image_seed = hashlib.sha256(
        json.dumps(image_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    parent_hash = image_seed
    chunks = []

    prompt_len = len(prompt_input_ids)
    chunk_bounds = []
    if min_reusable_prefix_len > 0:
        # Vision prompts need the post-image boundary even when it is not aligned
        # to the text chunk size. Plain text follows fixed full-size chunks.
        chunk_bounds.append(min(min_reusable_prefix_len, prompt_len))

    chunk_start = chunk_bounds[-1] if chunk_bounds else 0
    while chunk_start + chunk_size <= prompt_len:
        chunk_start += chunk_size
        chunk_bounds.append(chunk_start)

    previous_chunk_end = 0
    for chunk_end in chunk_bounds:
        payload = json.dumps(
            {
                "parent_hash": parent_hash,
                "chunk_tokens": prompt_input_ids[previous_chunk_end:chunk_end],
            },
            separators=(",", ":"),
        )
        parent_hash = hashlib.sha256(payload.encode()).hexdigest()
        if chunk_end >= min_reusable_prefix_len:
            chunks.append(
                PrefixCacheChunk(
                    start=previous_chunk_end,
                    end=chunk_end,
                    key=_make_chunk_key(previous_chunk_end, chunk_end, parent_hash),
                    chunk_hash=parent_hash,
                    image_hashes=list(image_hashes),
                )
            )
        previous_chunk_end = chunk_end

    return chunks


def _make_chunk_key(chunk_start: int, chunk_end: int, chunk_hash: str) -> str:
    payload = json.dumps(
        {
            "kind": "prompt_chunk",
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk_hash": chunk_hash,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_record_key(chunk_key: str, record_kind: str) -> str:
    payload = json.dumps(
        {
            "kind": "prompt_record",
            "chunk_key": chunk_key,
            "record_kind": record_kind,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_prefix_cache_chunks(
    prompt_input_ids: list[int],
    image_hashes: list[str],
    min_reusable_prefix_len: int,
) -> list[PrefixCacheChunk]:
    return _build_chunk_metadata(
        prompt_input_ids,
        image_hashes,
        min_reusable_prefix_len,
    )


def build_prefix_cache_boundaries(
    prompt_input_ids: list[int],
    image_hashes: list[str],
    min_reusable_prefix_len: int,
) -> list[int]:
    chunks = build_prefix_cache_chunks(
        prompt_input_ids,
        image_hashes,
        min_reusable_prefix_len,
    )
    max_reusable_prefix_len = len(prompt_input_ids) - 1
    return [chunk.end for chunk in chunks if 0 < chunk.end <= max_reusable_prefix_len]


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

        prompt_cache = _assemble_prompt_cache_chunks(
            chunk_prompt_caches,
            chunk_metadata,
        )
        # Restore can happen on a background thread; decode consumes the cache on
        # the scheduler thread. Force assembled arrays now so no lazy graph keeps
        # a thread-local MLX stream from the restore worker.
        _materialize_prompt_state(prompt_cache, rope_deltas)

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

        rope_deltas = _deserialize_rope_deltas(serialized_rope_deltas)

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
        final_metadata = self._metadata_by_key.get(chunk_keys[-1])
        if final_metadata is None:
            return None

        final_chunk_end = final_metadata.chunk_end
        final_chunk_key = chunk_keys[-1]
        rotating_window_size = self._rotating_window_size_for_sequence_locked(
            chunk_keys
        )
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

                record_key = _make_record_key(chunk_key, record_kind)
                if (
                    record_key not in self._record_metadata_by_key
                    or not self._store.exists(record_key)
                ):
                    return None
                record_keys.append(record_key)
            record_keys_by_chunk_key[chunk_key] = record_keys

        return record_keys_by_chunk_key

    def _rotating_window_size_for_sequence_locked(
        self, chunk_keys: list[str]
    ) -> Optional[int]:
        window_size = None
        for chunk_key in chunk_keys:
            record_key = _make_record_key(chunk_key, RECORD_KIND_ROTATING_DELTA)
            record_metadata = self._record_metadata_by_key.get(record_key)
            if record_metadata is not None and record_metadata.window_size is not None:
                window_size = max(window_size or 0, record_metadata.window_size)

        return window_size

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
        payload_cache, payload_kinds = _prepare_prompt_cache_payload(
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
        serialized_rope_deltas = _serialize_rope_deltas(rope_deltas)

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
        record_key = _make_record_key(chunk_key, record_kind)
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
        return [
            _make_record_key(chunk_key, record_kind)
            for record_kind in RECORD_WRITE_ORDER
            if record_kind in metadata.payload_kinds
        ]

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
        metadata = self._metadata_by_key.get(chunk_key)
        if metadata is None:
            return False

        return any(
            record_key in self._record_metadata_by_key
            for record_key in self._get_chunk_record_keys_locked(chunk_key, metadata)
        )

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
            if chunk_metadata is not None and self._has_dependent_chunks_locked(
                record_metadata.chunk_key, chunk_metadata
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
            stale = (
                self._is_stale_rotating_record_locked(
                    record_metadata,
                    chunk_metadata,
                )
                if record_metadata.record_kind == RECORD_KIND_ROTATING_DELTA
                else self._is_stale_state_checkpoint_record_locked(
                    record_metadata,
                    chunk_metadata,
                )
            )
            if stale:
                return key

        return None

    def _is_stale_rotating_record_locked(
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
                or not self._chunk_has_live_records_locked(candidate_key)
                or candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                != metadata.prompt_input_ids
            ):
                continue

            window_start = candidate_metadata.chunk_end - record_metadata.window_size
            if metadata.chunk_end > window_start:
                continue

            candidate_chunk_keys = [
                chunk.key
                for chunk in build_prefix_cache_chunks(
                    candidate_metadata.prompt_input_ids,
                    candidate_metadata.image_hashes,
                    candidate_metadata.min_reusable_prefix_len,
                )
                if chunk.end <= candidate_metadata.chunk_end
            ]
            if (
                candidate_chunk_keys
                and candidate_chunk_keys[-1] == candidate_key
                and self._record_keys_for_chunk_sequence_locked(candidate_chunk_keys)
                is not None
            ):
                return True

        return False

    def _is_stale_state_checkpoint_record_locked(
        self,
        record_metadata: CachedPromptRecordMetadata,
        metadata: CachedPromptMetadata,
    ) -> bool:
        for candidate_key, candidate_metadata in self._metadata_by_key.items():
            if (
                candidate_key == record_metadata.chunk_key
                or candidate_metadata.chunk_end <= metadata.chunk_end
                or candidate_metadata.image_hashes != metadata.image_hashes
                or not self._chunk_has_live_records_locked(candidate_key)
                or candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                != metadata.prompt_input_ids
            ):
                continue

            candidate_chunk_keys = [
                chunk.key
                for chunk in build_prefix_cache_chunks(
                    candidate_metadata.prompt_input_ids,
                    candidate_metadata.image_hashes,
                    candidate_metadata.min_reusable_prefix_len,
                )
                if chunk.end <= candidate_metadata.chunk_end
            ]
            if (
                candidate_chunk_keys
                and candidate_chunk_keys[-1] == candidate_key
                and self._record_keys_for_chunk_sequence_locked(candidate_chunk_keys)
                is not None
            ):
                return True

        return False

    def _has_dependent_chunks_locked(
        self,
        chunk_key: str,
        metadata: CachedPromptMetadata,
    ) -> bool:
        for candidate_key, candidate_metadata in self._metadata_by_key.items():
            if (
                candidate_key == chunk_key
                or candidate_metadata.chunk_end <= metadata.chunk_end
                or candidate_metadata.image_hashes != metadata.image_hashes
                or not self._chunk_has_live_records_locked(candidate_key)
            ):
                continue

            if (
                candidate_metadata.prompt_input_ids[: metadata.chunk_end]
                == metadata.prompt_input_ids
            ):
                return True

        return False

    def _evict_key(self, key: str) -> None:
        self._store.delete(key)

        with self._metadata_index_lock:
            self._total_bytes -= self._key_sizes.pop(key, 0)
            self._lru_keys.pop(key, None)
            self._cache_evictions += 1

        self._remove_record_metadata(key)
