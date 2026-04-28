from dataclasses import dataclass
import hashlib
import json
from typing import Any, Optional


# LMCache defaults to 256-token external chunks, and MLX KV caches allocate in
# 256-token steps. This is a spill-cache chunk size, not vLLM's KV page size.
DEFAULT_PREFIX_CHUNK_SIZE = 256


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


def make_record_key(chunk_key: str, record_kind: str) -> str:
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
