from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
    extend_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan


def _chunk_bounds(prompt_len: int, image_spans=None) -> list[tuple[int, int]]:
    chunks = build_prefix_cache_chunks(list(range(prompt_len)), image_spans or [])
    return [(chunk.start, chunk.end) for chunk in chunks]


def test_chunks_drop_short_tail():
    """Only full 256-token cache chunks are emitted; short tails stay hot."""
    assert _chunk_bounds(255) == []
    assert _chunk_bounds(511) == [(0, 256)]
    assert _chunk_bounds(512) == [(0, 256), (256, 512)]


def test_chunks_include_image_identity_without_growing_bounds():
    """Image spans crossing a boundary mark both fixed-size chunks."""
    image_spans = [PromptImageSpan(start=250, end=300, image_hash="image-a")]

    assert _chunk_bounds(600, image_spans) == [(0, 256), (256, 512)]

    changed_image_spans = [PromptImageSpan(start=250, end=300, image_hash="image-b")]
    chunks_a = build_prefix_cache_chunks(list(range(600)), image_spans)
    chunks_b = build_prefix_cache_chunks(list(range(600)), changed_image_spans)

    assert chunks_a[0].key != chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key


def test_chunks_later_images_do_not_invalidate_earlier_chunks():
    """Later image changes preserve earlier chunk keys and change later keys."""
    prompt_input_ids = list(range(900))
    image_a = [PromptImageSpan(start=300, end=320, image_hash="image-a")]
    image_b = [PromptImageSpan(start=300, end=320, image_hash="image-b")]

    chunks_a = build_prefix_cache_chunks(prompt_input_ids, image_a)
    chunks_b = build_prefix_cache_chunks(prompt_input_ids, image_b)

    assert chunks_a[0].key == chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key
    assert chunks_a[2].key != chunks_b[2].key


def test_chunks_later_tokens_do_not_invalidate_earlier_chunks():
    """Later token changes preserve earlier chunk keys and change later keys."""
    prompt_a = list(range(900))
    prompt_b = list(prompt_a)
    prompt_b[300] = -1

    chunks_a = build_prefix_cache_chunks(prompt_a, [])
    chunks_b = build_prefix_cache_chunks(prompt_b, [])

    assert chunks_a[0].key == chunks_b[0].key
    assert chunks_a[1].key != chunks_b[1].key
    assert chunks_a[2].key != chunks_b[2].key


def test_chunks_extend_incrementally_matches_full_build():
    """Decode can append only newly completed chunks without rebuilding all."""
    prompt_input_ids = list(range(700))
    image_spans = [PromptImageSpan(start=300, end=320, image_hash="image-a")]
    chunks = build_prefix_cache_chunks(prompt_input_ids[:300], image_spans)

    extend_prefix_cache_chunks(prompt_input_ids, image_spans, chunks)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 256), (256, 512)]
    assert chunks == build_prefix_cache_chunks(prompt_input_ids, image_spans)
