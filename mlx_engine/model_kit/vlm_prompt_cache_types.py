from dataclasses import dataclass
import hashlib
from typing import Any, Final, Literal, Optional, TypeAlias


# LMCache defaults to 256-token external chunks, and MLX KV caches allocate in
# 256-token steps. This is a spill-cache chunk size, not vLLM's KV page size.
DEFAULT_PREFIX_CHUNK_SIZE = 256
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
    cached_prefix_len: int
    prompt_cache: list[Any]


@dataclass
class PromptPrefixChunk:
    """Logical identity for one reusable prompt-prefix chunk.

    `key` is the rolling chunk hash. It includes prior chunks plus any image
    hashes whose placeholder spans are inside this chunk, so appending a later
    image does not invalidate earlier chunks. Physical safetensor blobs are
    keyed separately with `make_record_key(key, record_kind)`.
    """

    start: int
    end: int
    key: str
    chunk_hash: str


@dataclass
class PromptCacheLayout:
    """Stable prompt-cache layer layout for one model load."""

    layer_kinds: list[RecordKind]
    layer_indices_by_kind: dict[RecordKind, list[int]]
    rotating_window_size: Optional[int] = None


@dataclass
class PromptCacheRecordMetadata:
    """Index metadata for one physical safetensor record.

    A record stores one payload kind for one chunk, usually covering one or more
    cache layers.
    """

    chunk_key: str
    record_kind: RecordKind
    layer_indices: list[int]


@dataclass
class PreparedPromptRecord:
    key: str
    metadata: PromptCacheRecordMetadata
    snapshot_arrays: dict[str, Any]
    safetensor_metadata: dict[str, str]


@dataclass
class PendingPromptCacheSave:
    """Prepared cache-boundary save awaiting actor-thread disk commit/discard."""

    prefix_chunks: list[PromptPrefixChunk]
    cache_layout: PromptCacheLayout
    records: list[PreparedPromptRecord]


@dataclass
class VlmPromptSpillCacheStats:
    """Committed spill-cache accounting used by diagnostics and smokes.

    `hits` count restored chunks. `misses` count eligible chunks that could
    not be restored.
    """

    total_bytes: int
    max_bytes: int
    entry_count: int
    hits: int
    misses: int
    evictions: int
    record_sizes: list[int]
    record_sizes_by_key: dict[str, int]
    chunk_sizes_by_key: dict[str, int]
    chunk_records_available_by_key: dict[str, bool]


def make_record_key(chunk_key: str, record_kind: RecordKind) -> str:
    return f"record:{chunk_key}:{record_kind}"


def record_kind_for_prompt_cache(cache: Any) -> RecordKind:
    """Classify one live prompt-cache layer into its disk record kind."""
    cache_type = type(cache).__name__
    if cache_type == "KVCache":
        # mlx-vlm re-exports mlx-lm cache classes; keep this name-based so local
        # forks do not need identical module identities.
        return RECORD_KIND_KV_DELTA
    if cache_type == "RotatingKVCache" and getattr(cache, "keep", 0) == 0:
        return RECORD_KIND_ROTATING_DELTA
    return RECORD_KIND_STATE_CHECKPOINT


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
    chunk_bounds = _build_prefix_cache_chunk_bounds(
        len(prompt_input_ids),
        image_spans,
        chunk_size,
    )
    prefix_hash = hashlib.sha256(b"prompt-prefix-v1").hexdigest()
    chunks = []

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
        chunks.append(
            PromptPrefixChunk(
                start=previous_chunk_end,
                end=chunk_end,
                key=chunk_key,
                chunk_hash=prefix_hash,
            )
        )
        previous_chunk_end = chunk_end

    return chunks


def _build_prefix_cache_chunk_bounds(
    prompt_len: int,
    image_spans: list[PromptImageSpan],
    chunk_size: int,
) -> list[int]:
    """Return disk-save chunk ends, treating chunk_size as a minimum."""
    sorted_image_spans = sorted(image_spans, key=lambda item: item.start)
    chunk_bounds = []
    cursor = 0
    while cursor + chunk_size <= prompt_len:
        chunk_end = cursor + chunk_size
        for span in sorted_image_spans:
            if chunk_end <= span.start:
                break
            if span.start < chunk_end < span.end:
                # Keep image placeholders atomic by growing this disk chunk.
                chunk_end = span.end
                break
        chunk_bounds.append(chunk_end)
        cursor = chunk_end

    return chunk_bounds


def build_prefix_cache_save_points_for_length(
    prompt_len: int,
    image_spans: list[PromptImageSpan],
    chunk_size: int = DEFAULT_PREFIX_CHUNK_SIZE,
) -> list[int]:
    """Return save points up to a planned prefix length.

    Decode-time save scheduling only needs boundary offsets; the actual token
    values are hashed later when the snapshot is prepared.
    """
    max_reusable_prefix_len = prompt_len - 1
    return [
        chunk_end
        for chunk_end in _build_prefix_cache_chunk_bounds(
            prompt_len,
            image_spans,
            chunk_size,
        )
        if 0 < chunk_end <= max_reusable_prefix_len
    ]
