"""Fail fast on checkpoint layouts that the MLX engine cannot load."""

import json
from pathlib import Path

from safetensors import safe_open

# mlx-lm saves Inkling 2 Mini (`inkling2_loop`) with tensor names under
# `model.language_model.`. mlx-vlm, which loads all vision models here, only
# reads the original checkpoint names and its own converted names, so it would
# otherwise fail with a long list of unrecognized parameters.
_MLX_LM_INKLING2_PREFIX = "model.language_model."
UNSUPPORTED_INKLING2_MESSAGE = (
    "Unsupported Inkling 2 Mini format, try lmstudio-community/Inkling-2-Mini-MLX-4bit."
)


def _weight_names(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return list(json.loads(index_path.read_text())["weight_map"])
    names = []
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(path, framework="numpy") as file:
            names.extend(file.keys())
    return names


def raise_if_unsupported_checkpoint(model_path: Path, config_json: dict) -> None:
    if config_json.get("model_type") != "inkling2_loop":
        return
    if any(
        name.startswith(_MLX_LM_INKLING2_PREFIX) for name in _weight_names(model_path)
    ):
        raise ValueError(UNSUPPORTED_INKLING2_MESSAGE)
