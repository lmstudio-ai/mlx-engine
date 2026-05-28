from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias


# LMCache defaults to 256-token external chunks, and MLX KV caches allocate in
# 256-token steps. This is a cache store chunk size, not vLLM's KV page size.
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
    """End-exclusive token span for one image placeholder run.

    Span lists are sorted by start and non-overlapping.
    """

    start: int
    end: int
    image_hash: str


@dataclass
class LoadedDiskPromptCache:
    """Prompt cache materialized from physical disk cache records."""

    cached_prefix_len: int
    prompt_cache: list[Any]


@dataclass
class PromptPrefixChunk:
    """Logical identity for one reusable prompt-prefix chunk.

    `key` is the rolling chunk hash. It includes prior chunks plus image spans
    that overlap this chunk, so appending a later image does not invalidate
    earlier chunks. Physical safetensor blobs are keyed separately with
    `make_record_key(key, record_kind)`.
    """

    start: int
    end: int
    key: str


@dataclass
class PromptCacheLayout:
    """Stable prompt-cache layer layout for one model load."""

    layer_kinds: list[RecordKind]
    layer_indices_by_kind: dict[RecordKind, list[int]]
    rotating_window_size: int | None = None


@dataclass
class PromptCacheRecordMetadata:
    """Index metadata for one physical safetensor record.

    A record stores one record kind for one chunk, usually covering one or more
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
    """Prepared cache-boundary save awaiting cache-I/O-thread commit/discard."""

    prefix_chunks: list[PromptPrefixChunk]
    cache_layout: PromptCacheLayout
    records: list[PreparedPromptRecord]


@dataclass
class PromptCacheStoreStats:
    """Committed cache-store diagnostics used by smokes/debug output.

    Hit/miss tokens are coordinator-recorded restore accounting across both hot
    memory and disk.
    """

    total_bytes: int
    max_bytes: int
    entry_count: int
    hit_tokens: int
    miss_tokens: int
    evictions: int
    record_sizes: list[int]
    record_sizes_by_key: dict[str, int]
    chunk_sizes_by_key: dict[str, int]
    chunk_records_available_by_key: dict[str, bool]


def make_record_key(chunk_key: str, record_kind: RecordKind) -> str:
    return f"record:{chunk_key}:{record_kind}"
