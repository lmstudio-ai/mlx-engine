"""Shared helpers for patched-model parity tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

import mlx.core as mx
import mlx_lm.utils
from mlx_lm.models.cache import make_prompt_cache

import mlx_engine.model_kit  # noqa: F401
from mlx_vlm.utils import load_model as vlm_load_model
from tests.shared import model_getter


def get_real_model_path(model_name: str) -> Path:
    model_path = model_getter(model_name)
    if not any(model_path.glob("*.safetensors")):
        pytest.skip(f"{model_name}: no local MLX safetensors found in {model_path}")
    return model_path


def max_abs_diff(actual: mx.array, reference: mx.array) -> float:
    return float(mx.max(mx.abs(actual - reference)).item())


@contextmanager
def _temporary_bindings(module, **replacements) -> Iterator[None]:
    current_bindings = {name: getattr(module, name) for name in replacements}
    for name, replacement in replacements.items():
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for name, current_binding in current_bindings.items():
            setattr(module, name, current_binding)


def load_unpatched_mlx_lm(
    model_path: Path, *, module, original_bindings: dict[str, object]
):
    for name, original in original_bindings.items():
        if getattr(module, name) is original:
            raise AssertionError(
                f"Expected a pristine {module.__name__}.{name} reference captured "
                "before mlx-engine patched mlx-lm."
            )
    with _temporary_bindings(module, **original_bindings):
        return mlx_lm.utils.load(model_path)


def load_patched_mlx_lm(model_path: Path):
    return mlx_lm.utils.load(model_path)


def load_vlm(model_path: Path):
    result = vlm_load_model(model_path)
    return result[0] if isinstance(result, tuple) else result


def first_mlx_lm_generation_logits(
    model,
    prompt_tokens: mx.array,
    *,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
) -> mx.array:
    """Return the first-step logits from mlx-lm's generation path."""
    prompt_cache = make_prompt_cache(model)
    remaining_tokens = prompt_tokens
    remaining_embeddings = input_embeddings

    while len(remaining_tokens) > 1:
        n_to_process = min(prefill_step_size, len(remaining_tokens) - 1)
        if n_to_process <= 0:
            break
        kwargs = {"cache": prompt_cache}
        if remaining_embeddings is not None:
            kwargs["input_embeddings"] = remaining_embeddings[:n_to_process][None]
        model(remaining_tokens[:n_to_process][None], **kwargs)
        mx.eval([cache.state for cache in prompt_cache])
        remaining_tokens = remaining_tokens[n_to_process:]
        if remaining_embeddings is not None:
            remaining_embeddings = remaining_embeddings[n_to_process:]
        mx.clear_cache()

    kwargs = {"cache": prompt_cache}
    if remaining_embeddings is not None:
        kwargs["input_embeddings"] = remaining_embeddings[None]
    logits = model(remaining_tokens[None], **kwargs)
    mx.eval(logits)
    return mx.array(logits[0, -1, :])
