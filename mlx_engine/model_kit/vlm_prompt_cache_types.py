from dataclasses import dataclass
import hashlib
from typing import Any, Final, Literal, Optional, TypeAlias


# LMCache defaults to 256-token external chunks, and MLX KV caches allocate in
# 256-token steps. This is a spill-cache chunk size, not vLLM's KV page size.
DEFAULT_PREFIX_CHUNK_SIZE = 256
PrefixCacheSavePoint: TypeAlias = int
RecordKind: TypeAlias = Literal["kv_delta", "rotating_delta", "state_checkpoint"]
RECORD_KIND_KV_DELTA: Final[RecordKind] = "kv_delta"
RECORD_KIND_ROTATING_DELTA: Final[RecordKind] = "rotating_delta"
RECORD_KIND_STATE_CHECKPOINT: Final[RecordKind] = "state_checkpoint"
RECORD_WRITE_ORDER: Final[tuple[RecordKind, ...]] = (
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
)


@dataclass
class PromptImageSpan:
    """End-exclusive token span for one image placeholder run in the prompt."""

    start: int
    end: int
    image_hash: str


@dataclass
class SpilledPromptState:
    prompt_cache: list[Any]
    rope_deltas: Optional[Any]


@dataclass
class PromptPrefixChunk:
    """Logical identity for one reusable prompt-prefix chunk.

    `key` is the rolling chunk hash. It includes prior chunks plus any image
    hashes whose placeholder spans are inside this chunk, so appending a later
    image does not invalidate earlier chunks. Physical safetensor blobs are
    keyed separately with `make_record_key(key, record_kind)`.
    `prefix_chunk_keys` is the ordered restore chain through this chunk.
    """

    start: int
    end: int
    key: str
    chunk_hash: str
    prefix_chunk_keys: list[str]


@dataclass
class PromptCacheChunkMetadata:
    """Index metadata for one logical prompt-prefix chunk.

    This describes the prefix identity, its ordered chunk ancestry, and the
    per-layer payload kinds available for the chunk. Physical safetensor records
    are tracked by `PromptCacheRecordMetadata`.
    """

    chunk_start: int
    chunk_end: int
    chunk_hash: str
    prefix_chunk_keys: list[str]
    payload_kinds: list[RecordKind]


@dataclass
class PromptCacheRecordMetadata:
    """Index metadata for one physical safetensor record.

    A record stores one payload kind for one chunk, usually covering one or more
    cache layers. `window_size` is set only for rotating/sliding-window records.
    """

    chunk_key: str
    record_kind: RecordKind
    layer_indices: list[int]
    window_size: Optional[int] = None


@dataclass
class CachedPrefixMatch:
    key: str
    metadata: PromptCacheChunkMetadata
    matched_prefix_len: int
    chunk_keys: list[str]


@dataclass
class PreparedPromptRecord:
    key: str
    metadata: PromptCacheRecordMetadata
    snapshot_arrays: dict[str, Any]
    snapshot_metadata: dict[str, str]


@dataclass
class PendingPromptCacheSave:
    """Prepared cache-boundary save awaiting actor-thread disk commit/discard."""

    key: str
    metadata: PromptCacheChunkMetadata
    rope_deltas: Optional[Any]
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


def make_record_key(chunk_key: str, record_kind: RecordKind) -> str:
    return f"record:{chunk_key}:{record_kind}"


def _image_fingerprint_for_chunk(
    image_spans: list[PromptImageSpan],
    chunk_start: int,
    chunk_end: int,
) -> str:
    """Return image identities contained in this chunk.

    Prompt token ids only contain image placeholders, so the chunk hash must
    also include the image hash. Image spans are atomic chunks, so position and
    length are already implied by the chunk's tokens and bounds.
    """
    image_hashes = []
    for span in image_spans:
        if chunk_start <= span.start and span.end <= chunk_end:
            image_hashes.append(span.image_hash)
    return ",".join(image_hashes)


def build_prefix_cache_chunks(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    chunk_size: int = DEFAULT_PREFIX_CHUNK_SIZE,
) -> list[PromptPrefixChunk]:
    prefix_hash = hashlib.sha256(b"prompt-prefix-v1").hexdigest()
    chunks = []
    prefix_chunk_keys: list[str] = []

    prompt_len = len(prompt_input_ids)
    chunk_bounds = []
    cursor = 0
    for span in sorted(image_spans, key=lambda item: item.start):
        while cursor + chunk_size <= span.start:
            cursor += chunk_size
            chunk_bounds.append(cursor)
        if cursor < span.start:
            # Image spans are atomic, but text before them can still be cached.
            chunk_bounds.append(span.start)
        if chunk_bounds and chunk_bounds[-1] == span.start:
            cursor = span.start
        cursor = max(cursor, span.end)
        chunk_bounds.append(cursor)

    while cursor + chunk_size <= prompt_len:
        cursor += chunk_size
        chunk_bounds.append(cursor)

    previous_chunk_end = 0
    for chunk_end in chunk_bounds:
        if chunk_end <= previous_chunk_end:
            continue
        chunk_tokens = prompt_input_ids[previous_chunk_end:chunk_end]
        image_fingerprint = _image_fingerprint_for_chunk(
            image_spans,
            previous_chunk_end,
            chunk_end,
        )
        payload = (
            f"{prefix_hash}|{','.join(map(str, chunk_tokens))}|{image_fingerprint}"
        )
        prefix_hash = hashlib.sha256(payload.encode()).hexdigest()
        chunk_key = prefix_hash
        prefix_chunk_keys.append(chunk_key)
        chunks.append(
            PromptPrefixChunk(
                start=previous_chunk_end,
                end=chunk_end,
                key=chunk_key,
                chunk_hash=prefix_hash,
                prefix_chunk_keys=list(prefix_chunk_keys),
            )
        )
        previous_chunk_end = chunk_end

    return chunks


def build_prefix_cache_save_points(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
) -> list[PrefixCacheSavePoint]:
    """Return end-exclusive prompt prefix lengths where cache should be saved."""
    chunks = build_prefix_cache_chunks(
        prompt_input_ids,
        image_spans,
    )
    max_reusable_prefix_len = len(prompt_input_ids) - 1
    return [chunk.end for chunk in chunks if 0 < chunk.end <= max_reusable_prefix_len]
