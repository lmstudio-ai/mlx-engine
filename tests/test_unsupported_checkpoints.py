import json
import struct
from pathlib import Path

import pytest

import mlx_engine.generate as generate_module
from mlx_engine.utils.unsupported_checkpoints import (
    UNSUPPORTED_INKLING2_MESSAGE,
    raise_if_unsupported_checkpoint,
)

MLX_LM_INKLING2_NAME = "model.language_model.layers.0.mlp.experts.gate_up_proj.weight"
MLX_VLM_INKLING2_NAME = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
ORIGINAL_INKLING2_NAME = "model.llm.layers.0.mlp.experts.w13_weight"


def _write_checkpoint(tmp_path: Path, model_type: str, weight_name: str) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: "model-00001-of-00001.safetensors"}})
    )
    return tmp_path


def _write_single_file_checkpoint(tmp_path: Path, weight_name: str) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
    header = json.dumps(
        {
            "__metadata__": {"format": "mlx"},
            weight_name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        }
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00\x00"
    )
    return tmp_path


def test_rejects_mlx_lm_inkling2_checkpoint(tmp_path):
    model_path = _write_checkpoint(tmp_path, "inkling2_loop", MLX_LM_INKLING2_NAME)

    with pytest.raises(ValueError, match=UNSUPPORTED_INKLING2_MESSAGE):
        raise_if_unsupported_checkpoint(model_path, {"model_type": "inkling2_loop"})


def test_rejects_single_file_mlx_lm_inkling2_checkpoint(tmp_path):
    model_path = _write_single_file_checkpoint(tmp_path, MLX_LM_INKLING2_NAME)

    with pytest.raises(ValueError, match=UNSUPPORTED_INKLING2_MESSAGE):
        raise_if_unsupported_checkpoint(model_path, {"model_type": "inkling2_loop"})


@pytest.mark.parametrize(
    "model_type, weight_name",
    [
        ("inkling2_loop", MLX_VLM_INKLING2_NAME),
        ("inkling2_loop", ORIGINAL_INKLING2_NAME),
        ("qwen3_vl", MLX_LM_INKLING2_NAME),
    ],
)
def test_allows_supported_checkpoints(tmp_path, model_type, weight_name):
    model_path = _write_checkpoint(tmp_path, model_type, weight_name)

    raise_if_unsupported_checkpoint(model_path, {"model_type": model_type})


def test_load_model_rejects_before_creating_a_kit(monkeypatch, tmp_path):
    model_path = _write_checkpoint(tmp_path, "inkling2_loop", MLX_LM_INKLING2_NAME)

    def fail_if_created(*_args, **_kwargs):
        raise AssertionError("No model kit should be created")

    monkeypatch.setattr(generate_module, "BatchedVisionModelKit", fail_if_created)
    monkeypatch.setattr(generate_module, "ModelKit", fail_if_created)

    with pytest.raises(ValueError, match=UNSUPPORTED_INKLING2_MESSAGE):
        generate_module.load_model(model_path)
