"""Prompt chunking helpers.

A chunk is one reusable slice of a prompt prefix. We save cache records per
chunk, then restore by replaying the ordered chunk chain whose tokens/images
match the next request.
"""

import hashlib

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptImageSpan,
    PromptPrefixChunk,
)

_INITIAL_PREFIX_HASH = hashlib.sha256(b"prompt-prefix-v1").hexdigest()


def build_prefix_cache_chunks(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
) -> list[PromptPrefixChunk]:
    """Return fixed-size rolling-hash prompt chunks."""
    chunks = []
    extend_prefix_cache_chunks(prompt_input_ids, image_spans, chunks)
    return chunks


def extend_prefix_cache_chunks(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    chunks: list[PromptPrefixChunk],
) -> None:
    """Append newly complete 256-token chunks in place."""
    chunk_start = chunks[-1].end if chunks else 0
    previous_chunk_key = chunks[-1].key if chunks else _INITIAL_PREFIX_HASH

    while chunk_start + DEFAULT_PREFIX_CHUNK_SIZE <= len(prompt_input_ids):
        chunk_end = chunk_start + DEFAULT_PREFIX_CHUNK_SIZE
        chunk = _make_prefix_cache_chunk(
            previous_chunk_key,
            prompt_input_ids,
            image_spans,
            chunk_start,
            chunk_end,
        )
        chunks.append(chunk)
        previous_chunk_key = chunk.key
        chunk_start = chunk_end


def first_unsaved_prefix_cache_chunk_index(
    chunks: list[PromptPrefixChunk],
    prompt_progress: int,
) -> int:
    """Return the first chunk ending after the already-cached prefix."""
    return min(prompt_progress // DEFAULT_PREFIX_CHUNK_SIZE, len(chunks))


def _make_prefix_cache_chunk(
    previous_chunk_key: str,
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    chunk_start: int,
    chunk_end: int,
) -> PromptPrefixChunk:
    chunk_tokens = prompt_input_ids[chunk_start:chunk_end]
    image_fingerprint = _image_fingerprint_for_chunk(
        image_spans,
        chunk_start,
        chunk_end,
    )
    hash_input = (
        f"{previous_chunk_key}|{','.join(map(str, chunk_tokens))}|{image_fingerprint}"
    )
    chunk_key = hashlib.sha256(hash_input.encode()).hexdigest()
    return PromptPrefixChunk(
        start=chunk_start,
        end=chunk_end,
        key=chunk_key,
    )


def _image_fingerprint_for_chunk(
    image_spans: list[PromptImageSpan],
    chunk_start: int,
    chunk_end: int,
) -> str:
    """Return image identities whose placeholder spans overlap this chunk.

    Token ids locate image placeholders; this only adds image content identity.
    """
    image_fingerprints = []
    for span in image_spans:
        if span.start >= chunk_end:
            break
        if span.start < chunk_end and chunk_start < span.end:
            image_fingerprints.append(span.image_hash)
    return ",".join(image_fingerprints)
