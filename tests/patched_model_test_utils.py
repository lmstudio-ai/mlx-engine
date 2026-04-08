"""Shared helpers for patched-model parity tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

import mlx.core as mx
import mlx_lm.models.gemma4_text as gemma4_text_module
import mlx_lm.models.qwen3_5 as qwen3_5_module
import mlx_lm.utils
from mlx_lm.models.cache import make_prompt_cache

import mlx_engine.model_kit  # noqa: F401
from mlx_engine.model_kit.patches.gemma4 import PatchedGemma4TextModel
from mlx_engine.model_kit.patches.qwen3_5 import (
    OriginalDecoderLayer,
    OriginalQwen3_5TextModel,
)
from mlx_vlm.models.cache import make_prompt_cache as make_vlm_prompt_cache
from mlx_vlm.utils import load_model as vlm_load_model, load_processor
from tests.shared import model_getter
from transformers import AutoProcessor

OriginalGemma4TextModel = PatchedGemma4TextModel.__mro__[1]

REAL_MODEL_CASES = [
    pytest.param("lmstudio-community/Qwen3.5-2B-MLX-4bit", id="dense"),
    pytest.param(
        "lmstudio-community/Qwen3.5-35B-A3B-MLX-4bit",
        marks=pytest.mark.heavy,
        id="moe",
    ),
]
GEMMA4_MODEL_NAME = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
GEMMA4_IMAGE_TOPK = 5
GEMMA4_IMAGE_TOPK_PROB_RTOL = 0.25
GEMMA4_IMAGE_TOPK_PROB_REF_FLOOR = 1e-3


def get_real_model_path(model_name: str) -> Path:
    model_path = model_getter(model_name)
    if not any(model_path.glob("*.safetensors")):
        pytest.skip(f"{model_name}: no local MLX safetensors found in {model_path}")
    return model_path


def max_abs_diff(actual: mx.array, reference: mx.array) -> float:
    return float(mx.max(mx.abs(actual - reference)).item())


def tokenize_prompt(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    return [ids] if isinstance(ids, int) else ids


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


def _assert_restorable_binding(original, current, label: str) -> None:
    if original is current:
        raise AssertionError(
            f"Expected a pristine {label} reference captured before mlx-engine "
            "patched mlx-lm."
        )


def _load_unpatched_mlx_lm(
    model_path: Path, *, module, replacements: dict[str, object]
):
    with _temporary_bindings(module, **replacements):
        return mlx_lm.utils.load(model_path)


def load_patched_mlx_lm(model_path: Path):
    return mlx_lm.utils.load(model_path)


def load_unpatched_qwen_mlx_lm(model_path: Path):
    _assert_restorable_binding(
        OriginalDecoderLayer,
        qwen3_5_module.DecoderLayer,
        "qwen3.5 DecoderLayer",
    )
    _assert_restorable_binding(
        OriginalQwen3_5TextModel,
        qwen3_5_module.Qwen3_5TextModel,
        "qwen3.5 Qwen3_5TextModel",
    )
    return _load_unpatched_mlx_lm(
        model_path,
        module=qwen3_5_module,
        replacements={
            "DecoderLayer": OriginalDecoderLayer,
            "Qwen3_5TextModel": OriginalQwen3_5TextModel,
        },
    )


def load_unpatched_gemma4_mlx_lm(model_path: Path):
    _assert_restorable_binding(
        OriginalGemma4TextModel,
        gemma4_text_module.Gemma4TextModel,
        "Gemma4TextModel",
    )
    return _load_unpatched_mlx_lm(
        model_path,
        module=gemma4_text_module,
        replacements={"Gemma4TextModel": OriginalGemma4TextModel},
    )


def load_vlm(model_path: Path):
    result = vlm_load_model(model_path)
    return result[0] if isinstance(result, tuple) else result


def load_vlm_processor(model_path: Path):
    return load_processor(model_path, add_detokenizer=True)


def build_gemma4_prompt(
    model_path: Path,
    user_text: str,
    *,
    image_b64: str | None = None,
) -> str:
    processor = AutoProcessor.from_pretrained(model_path)
    content = [{"type": "text", "text": user_text}]
    if image_b64 is not None:
        content.insert(0, {"type": "image", "base64": image_b64})
    conversation = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


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


def first_vlm_generation_logits(
    model,
    *,
    input_ids: mx.array,
    pixel_values: mx.array,
    attention_mask: mx.array,
    prefill_step_size: int = 2048,
) -> mx.array:
    """Return the first-step logits from mlx-vlm's generation path."""
    prompt_cache = make_vlm_prompt_cache(model.language_model)
    embedding_output = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=pixel_values,
        mask=attention_mask,
    )
    inputs_embeds = embedding_output.inputs_embeds
    kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if key != "inputs_embeds" and value is not None
    }

    while inputs_embeds.shape[1] > 1:
        n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
        if n_to_process <= 0:
            break
        model.language_model(
            inputs=input_ids[:, :n_to_process],
            inputs_embeds=inputs_embeds[:, :n_to_process],
            cache=prompt_cache,
            n_to_process=n_to_process,
            **kwargs,
        )
        mx.eval([cache.state for cache in prompt_cache])
        input_ids = input_ids[:, n_to_process:]
        inputs_embeds = inputs_embeds[:, n_to_process:]
        mx.clear_cache()

    outputs = model.language_model(
        input_ids[:, -1:],
        inputs_embeds=inputs_embeds[:, -1:],
        cache=prompt_cache,
        **kwargs,
    )
    mx.eval(outputs.logits)
    return mx.array(outputs.logits[0, -1, :])


def topk_token_ids(logits: mx.array, k: int) -> list[int]:
    values = np.array(logits.tolist(), dtype=np.float32)
    return [int(index) for index in np.argsort(values)[-k:][::-1]]


def gather_values(values: mx.array, token_ids: list[int]) -> list[float]:
    return [float(values[token_id].item()) for token_id in token_ids]


def softmax_probabilities(logits: mx.array) -> mx.array:
    return mx.softmax(logits.astype(mx.float32), axis=-1)


def relative_differences(
    actual_values: list[float],
    reference_values: list[float],
    reference_floor: float,
) -> list[float]:
    diffs = []
    for actual, reference in zip(actual_values, reference_values):
        scale = max(abs(reference), reference_floor)
        diffs.append(abs(actual - reference) / scale)
    return diffs


def format_token_values(token_ids: list[int], values: list[float], tokenizer) -> str:
    parts = []
    for token_id, value in zip(token_ids, values):
        parts.append(f"{token_id}:{tokenizer.decode([token_id])!r}:{value:.6f}")
    return "[" + ", ".join(parts) + "]"


def resolve_image_token_index(config) -> int | None:
    vision_config = getattr(config, "vision_config", None)
    return getattr(
        config,
        "image_token_index",
        getattr(
            config,
            "image_token_id",
            getattr(vision_config, "image_token_id", None),
        ),
    )
