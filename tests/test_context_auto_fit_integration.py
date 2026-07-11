import json
from types import SimpleNamespace

import pytest

import mlx_engine.generate as generate_module


class _FakeVisionModelKit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


@pytest.mark.parametrize("auto_fit_context", [False, True])
def test_load_model_forwards_context_auto_fit_to_batched_vision(
    monkeypatch,
    tmp_path,
    auto_fit_context,
):
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": {}}))
    monkeypatch.setattr(
        generate_module,
        "BatchedVisionModelKit",
        _FakeVisionModelKit,
    )
    monkeypatch.setattr(generate_module, "sanitize_eos_tokens", lambda _kit: None)

    model_kit = generate_module.load_model(
        tmp_path,
        max_kv_size=12_345,
        auto_fit_context=auto_fit_context,
    )

    assert model_kit.started
    assert model_kit.kwargs["max_kv_size"] == 12_345
    assert model_kit.kwargs["auto_fit_context"] is auto_fit_context


def test_runtime_load_info_reports_only_fitted_context():
    fitted_kit = SimpleNamespace(effective_context_length=65_536)
    ordinary_kit = SimpleNamespace()

    assert generate_module.get_runtime_load_info(fitted_kit) == {
        "context_length": 65_536
    }
    assert generate_module.get_runtime_load_info(ordinary_kit) == {}
