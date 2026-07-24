import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mlx_engine.generate as generate_module


class _MergeableCache:
    def merge(self, caches):
        return caches


class _NonMergeableCache:
    pass


def _write_text_config(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "max_position_embeddings": 32_768,
                "num_attention_heads": 8,
            }
        )
    )
    return tmp_path


def _install_fake_kit(monkeypatch, class_name: str):
    created_kits = []

    class FakeKit:
        def __init__(self, model_path, **options):
            self.model_path = model_path
            self.options = options
            self.started = False
            created_kits.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(generate_module, class_name, FakeKit)
    return FakeKit, created_kits


def _disable_eos_sanitization(monkeypatch) -> None:
    monkeypatch.setattr(generate_module, "sanitize_eos_tokens", lambda _model_kit: None)


def test_batchable_text_model_uses_mlx_vlm_kit(monkeypatch, tmp_path):
    model_path = _write_text_config(tmp_path)
    fake_kit_class, created_kits = _install_fake_kit(
        monkeypatch,
        "BatchedVisionModelKit",
    )
    loaded_models = []

    def fake_load_model(path, *, lazy):
        loaded_models.append((path, lazy))
        return SimpleNamespace(language_model=SimpleNamespace())

    monkeypatch.setattr(generate_module, "mlx_vlm_load_model", fake_load_model)
    monkeypatch.setattr(
        generate_module,
        "make_prompt_cache",
        lambda _language_model: [_MergeableCache()],
    )
    _disable_eos_sanitization(monkeypatch)

    model_kit = generate_module.load_model(
        model_path,
        max_kv_size=8_192,
        max_seq_nums=3,
        trust_remote_code=True,
        seed=7,
    )

    assert isinstance(model_kit, fake_kit_class)
    assert model_kit is created_kits[0]
    assert model_kit.started
    assert loaded_models == [(model_path, True)]
    assert model_kit.options == {
        "max_kv_size": 8_192,
        "max_seq_nums": 3,
        "prefill_step_size": 2_048,
        "trust_remote_code": True,
        "seed": 7,
    }


def test_non_mergeable_text_model_stays_sequential(monkeypatch, tmp_path):
    model_path = _write_text_config(tmp_path)
    fake_kit_class, _created_kits = _install_fake_kit(monkeypatch, "ModelKit")
    monkeypatch.setattr(
        generate_module,
        "mlx_vlm_load_model",
        lambda _path, *, lazy: SimpleNamespace(language_model=SimpleNamespace()),
    )
    monkeypatch.setattr(
        generate_module,
        "make_prompt_cache",
        lambda _language_model: [_NonMergeableCache()],
    )
    _disable_eos_sanitization(monkeypatch)

    model_kit = generate_module.load_model(model_path, max_seq_nums=4)

    assert isinstance(model_kit, fake_kit_class)
    assert model_kit.started


@pytest.mark.parametrize(
    "load_options",
    [
        pytest.param({"vocab_only": True}, id="vocab-only"),
        pytest.param({"max_seq_nums": 1}, id="single-sequence"),
        pytest.param({"kv_bits": 8, "kv_group_size": 64}, id="quantized-kv"),
    ],
)
def test_special_text_modes_stay_sequential(
    monkeypatch,
    tmp_path,
    load_options,
):
    model_path = _write_text_config(tmp_path)
    fake_kit_class, _created_kits = _install_fake_kit(monkeypatch, "ModelKit")

    def unexpected_vlm_load(*_arguments, **_options):
        pytest.fail("mlx-vlm preflight should not run for sequential text modes")

    monkeypatch.setattr(generate_module, "mlx_vlm_load_model", unexpected_vlm_load)
    _disable_eos_sanitization(monkeypatch)

    model_kit = generate_module.load_model(model_path, **load_options)

    assert isinstance(model_kit, fake_kit_class)
    assert model_kit.started


def test_vision_model_continues_to_use_mlx_vlm_kit(monkeypatch, tmp_path):
    model_path = _write_text_config(tmp_path)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 16},
            }
        )
    )
    fake_kit_class, _created_kits = _install_fake_kit(
        monkeypatch,
        "BatchedVisionModelKit",
    )

    def unexpected_vlm_preflight(*_arguments, **_options):
        pytest.fail("vision models should not use the text batching preflight")

    monkeypatch.setattr(
        generate_module,
        "mlx_vlm_load_model",
        unexpected_vlm_preflight,
    )
    _disable_eos_sanitization(monkeypatch)

    model_kit = generate_module.load_model(model_path)

    assert isinstance(model_kit, fake_kit_class)
    assert model_kit.started
