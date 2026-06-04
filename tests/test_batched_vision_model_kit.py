from types import SimpleNamespace

from mlx_engine.model_kit.batched_vision.model_kit import (
    _requires_global_no_chunked_prefill,
    _restore_splits_gemma4_image_span,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptImageSpan,
)


def test_global_no_chunked_prefill_exempts_only_gemma4_unified():
    model = SimpleNamespace(no_chunked_prefill=True)

    assert not _requires_global_no_chunked_prefill(model, "gemma4_unified")
    assert not _requires_global_no_chunked_prefill(model, "gemma4_unified_text")
    assert _requires_global_no_chunked_prefill(model, "gemma4")
    assert _requires_global_no_chunked_prefill(model, "gemma4_text")


def test_global_no_chunked_prefill_stays_disabled_without_model_flag():
    model = SimpleNamespace(no_chunked_prefill=False)

    assert not _requires_global_no_chunked_prefill(model, "gemma4")


def test_gemma4_restore_conflict_only_rejects_split_image_span():
    image_spans = [PromptImageSpan(start=200, end=700, image_hash="image")]

    assert not _restore_splits_gemma4_image_span(
        model_type="gemma4_unified",
        cached_prefix_len=199,
        image_spans=image_spans,
    )
    assert not _restore_splits_gemma4_image_span(
        model_type="gemma4_unified",
        cached_prefix_len=200,
        image_spans=image_spans,
    )
    assert _restore_splits_gemma4_image_span(
        model_type="gemma4_unified",
        cached_prefix_len=201,
        image_spans=image_spans,
    )
    assert not _restore_splits_gemma4_image_span(
        model_type="gemma4",
        cached_prefix_len=201,
        image_spans=image_spans,
    )
