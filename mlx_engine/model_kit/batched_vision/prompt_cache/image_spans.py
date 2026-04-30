from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan


def image_safe_common_prefix_len(
    prompt_input_ids: list[int],
    image_spans: list[PromptImageSpan],
    cached_input_ids: list[int],
    cached_image_spans: list[PromptImageSpan],
    *,
    max_prefix_len: int,
) -> int:
    """Return the token prefix that does not split or mismatch image spans."""
    common_prefix_len = 0
    for token, cached_token in zip(prompt_input_ids, cached_input_ids):
        if token != cached_token:
            break
        common_prefix_len += 1
    common_prefix_len = min(common_prefix_len, max_prefix_len)

    # Placeholder token ids can match while the underlying image differs.
    image_keys = {_image_span_key(span) for span in image_spans}
    cached_image_keys = {_image_span_key(span) for span in cached_image_spans}
    # Either prompt may contain a non-reusable image inside the prefix.
    non_reusable_starts = [
        # New prompt images must match images already present in the hot cache.
        _first_non_reusable_image_start(
            common_prefix_len,
            image_spans,
            cached_image_keys,
        ),
        # Hot-cache images must also still exist in the new prompt.
        _first_non_reusable_image_start(
            common_prefix_len,
            cached_image_spans,
            image_keys,
        ),
    ]
    non_reusable_starts = [start for start in non_reusable_starts if start is not None]
    return min(non_reusable_starts) if non_reusable_starts else common_prefix_len


def _image_span_key(span: PromptImageSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.image_hash


def _first_non_reusable_image_start(
    prefix_len: int,
    spans: list[PromptImageSpan],
    other_span_keys: set[tuple[int, int, str]],
) -> int | None:
    for span in spans:
        if span.start >= prefix_len:
            break
        if span.end > prefix_len:
            return span.start
        if _image_span_key(span) not in other_span_keys:
            return span.start
    return None
