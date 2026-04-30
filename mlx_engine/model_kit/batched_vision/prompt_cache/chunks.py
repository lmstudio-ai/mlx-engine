import hashlib

from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptImageSpan,
    PromptPrefixChunk,
)


def build_prefix_cache_chunks(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    chunk_size: int = DEFAULT_PREFIX_CHUNK_SIZE,
) -> list[PromptPrefixChunk]:
    """Return rolling-hash prompt chunks, growing chunks to avoid split images."""
    chunk_ends = _build_prefix_cache_chunk_ends(
        len(prompt_input_ids),
        image_spans,
        chunk_size,
    )
    prefix_hash = hashlib.sha256(b"prompt-prefix-v1").hexdigest()
    chunks = []

    previous_chunk_end = 0
    for chunk_end in chunk_ends:
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
            )
        )
        previous_chunk_end = chunk_end

    return chunks


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


def _build_prefix_cache_chunk_ends(
    prompt_len: int,
    image_spans: list[PromptImageSpan],
    chunk_size: int,
) -> list[int]:
    """Return disk-save chunk ends, treating chunk_size as a minimum."""
    sorted_image_spans = sorted(image_spans, key=lambda item: item.start)
    chunk_ends = []
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
        chunk_ends.append(chunk_end)
        cursor = chunk_end

    return chunk_ends
