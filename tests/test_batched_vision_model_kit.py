from queue import Queue
from types import SimpleNamespace

import pytest

from mlx_engine.model_kit.batched_vision.model_kit import (
    BatchedVisionModelKit,
    _requires_global_no_chunked_prefill,
    _restore_splits_gemma4_image_span,
)
import mlx_engine.model_kit.batched_vision.model_kit as model_kit_module
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


def test_load_model_forces_no_trust_remote_code(monkeypatch, tmp_path):
    loaded_model = SimpleNamespace()
    call = {}

    def fake_load_model(model_path, **kwargs):
        call["model_path"] = model_path
        call["kwargs"] = kwargs
        return loaded_model

    monkeypatch.setattr(
        model_kit_module.mlx_vlm.utils,
        "load_model",
        fake_load_model,
    )
    monkeypatch.setattr(
        model_kit_module,
        "patch_loaded_gemma4_model",
        lambda model: call.setdefault("patched_model", model),
    )
    monkeypatch.setattr(model_kit_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(model_kit_module.mx, "clear_cache", lambda: None)

    kit = object.__new__(BatchedVisionModelKit)
    kit._shutdown = SimpleNamespace(is_set=lambda: True)
    kit._model_path = tmp_path
    kit._trust_remote_code = True

    kit._load_model()

    assert kit.model is loaded_model
    assert call["model_path"] == tmp_path
    assert call["kwargs"] == {"lazy": False, "trust_remote_code": False}
    assert call["patched_model"] is loaded_model


def test_generate_fatal_error_preserves_state_for_exception_propagation(monkeypatch):
    class FakeBatchGenerator:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeController:
        instance = None

        def __init__(self, **kwargs):
            FakeController.instance = self
            self.cancelled = False

        def drain_generation_events(self, timeout):
            pass

        def insert_ready_requests(self):
            pass

        def admit_pending_requests(self):
            pass

        def step_generation(self):
            raise RuntimeError("decode failed")

        def cancel_all_requests(self):
            self.cancelled = True

    monkeypatch.setattr(
        model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(model_kit_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(model_kit_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(model_kit_module.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(model_kit_module.gc, "collect", lambda: None)
    monkeypatch.setattr(model_kit_module, "GenerationThreadController", FakeController)

    batch_generator = FakeBatchGenerator()
    kit = object.__new__(BatchedVisionModelKit)
    kit._seed = 0
    kit.model = SimpleNamespace()
    kit._shutdown = SimpleNamespace(is_set=lambda: False, set=lambda: None)
    kit._generation_thread = None
    kit._startup_complete = SimpleNamespace(set=lambda: None)
    kit._make_batch_generator = lambda: batch_generator
    kit._requests = Queue()
    kit._max_seq_nums = 1
    kit._cache_io_thread = SimpleNamespace(
        enqueue_restore=lambda request: None,
        close=lambda: None,
    )
    kit._insert_prepared_request = lambda *args: None
    kit._emit_response = lambda *args: None
    kit._finish_response = lambda *args: None
    kit._prompt_cache_coordinator = SimpleNamespace(clear_hot_prompt_cache=lambda: None)
    kit._vision_feature_memoizer = SimpleNamespace(clear=lambda: None)
    kit.processor = None
    kit.tokenizer = None
    kit.detokenizer = None

    with pytest.raises(RuntimeError, match="decode failed"):
        kit._generate()

    assert FakeController.instance is not None
    assert not FakeController.instance.cancelled
    assert batch_generator.closed
    assert kit._generation_thread_state is not None
