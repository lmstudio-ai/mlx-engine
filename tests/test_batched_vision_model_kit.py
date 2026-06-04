from types import SimpleNamespace

from mlx_engine.model_kit.batched_vision.model_kit import (
    _requires_global_no_chunked_prefill,
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
