from mlx_engine.model_kit.batched_vision.prompt_cache.image_spans import (
    image_safe_common_prefix_len,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan


def test_image_safe_prefix_stops_before_changed_image():
    """Matching placeholder tokens with different images must not reuse KV."""
    prompt_tokens = list(range(100))
    image_spans = [PromptImageSpan(20, 30, "new-image")]
    cached_image_spans = [PromptImageSpan(20, 30, "cached-image")]

    prefix_len = image_safe_common_prefix_len(
        prompt_tokens,
        image_spans,
        prompt_tokens,
        cached_image_spans,
        max_prefix_len=len(prompt_tokens),
    )

    assert prefix_len == 20


def test_image_safe_prefix_stops_before_partially_reused_image():
    """A reusable prefix must not split one expanded image span."""
    prompt_tokens = list(range(100))
    image_spans = [PromptImageSpan(20, 30, "image")]

    prefix_len = image_safe_common_prefix_len(
        prompt_tokens,
        image_spans,
        prompt_tokens,
        image_spans,
        max_prefix_len=25,
    )

    assert prefix_len == 20


def test_image_safe_prefix_stops_before_cached_image_missing_from_prompt():
    """Hot-cache images must still exist in the new prompt before reuse."""
    prompt_tokens = list(range(100))
    cached_image_spans = [PromptImageSpan(20, 30, "cached-image")]

    prefix_len = image_safe_common_prefix_len(
        prompt_tokens,
        [],
        prompt_tokens,
        cached_image_spans,
        max_prefix_len=len(prompt_tokens),
    )

    assert prefix_len == 20
