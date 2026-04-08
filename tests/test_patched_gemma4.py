"""Tests for the Gemma 4 monkey patch."""

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
import mlx_lm.models.gemma4_text as gemma4_text_module

from mlx_engine.generate import load_model
from mlx_engine.model_kit.patches.gemma4 import OriginalGemma4TextModel
from mlx_engine.utils.image_utils import convert_to_pil
from mlx_engine.utils.prompt_progress_reporter import DefaultPromptProgressReporter
from mlx_vlm.models.cache import make_prompt_cache as make_vlm_prompt_cache
from mlx_vlm.utils import load_processor, prepare_inputs

from tests.patched_model_test_utils import (
    first_mlx_lm_generation_logits,
    get_real_model_path,
    load_unpatched_mlx_lm,
    load_patched_mlx_lm,
    load_vlm,
    max_abs_diff,
)
from tests.shared import read_image_b64
from transformers import AutoProcessor

pytestmark = pytest.mark.heavy

GEMMA4_MODEL_NAME = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
GEMMA4_IMAGE_TOPK = 5
GEMMA4_IMAGE_TOPK_PROB_RTOL = 0.25
GEMMA4_IMAGE_TOPK_PROB_REF_FLOOR = 1e-3


def tokenize_prompt(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    return [ids] if isinstance(ids, int) else ids


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


def load_unpatched_gemma4_mlx_lm(model_path: Path):
    return load_unpatched_mlx_lm(
        model_path,
        module=gemma4_text_module,
        original_bindings={"Gemma4TextModel": OriginalGemma4TextModel},
    )


def load_vlm_processor(model_path: Path):
    return load_processor(model_path, add_detokenizer=True)


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


def test_gemma4_text_only_generation_patched_matches_unpatched():
    """The Gemma 4 patch must be a no-op for text-only generation."""
    model_path = get_real_model_path(GEMMA4_MODEL_NAME)
    prefill_step_size = 16
    user_text = " ".join(
        f"Segment {index}: explain why careful benchmarking matters before changing an inference stack."
        for index in range(1, 25)
    )
    prompt = build_gemma4_prompt(model_path, user_text)

    patched_model, patched_tokenizer = load_patched_mlx_lm(model_path)
    patched_prompt_tokens = tokenize_prompt(patched_tokenizer, prompt)
    assert len(patched_prompt_tokens) > prefill_step_size * 2
    patched_first_logits = first_mlx_lm_generation_logits(
        patched_model,
        mx.array(patched_prompt_tokens),
        prefill_step_size=prefill_step_size,
    )
    del patched_model
    del patched_tokenizer
    mx.clear_cache()

    unpatched_model, unpatched_tokenizer = load_unpatched_gemma4_mlx_lm(model_path)
    assert tokenize_prompt(unpatched_tokenizer, prompt) == patched_prompt_tokens
    unpatched_first_logits = first_mlx_lm_generation_logits(
        unpatched_model,
        mx.array(patched_prompt_tokens),
        prefill_step_size=prefill_step_size,
    )
    del unpatched_model
    del unpatched_tokenizer
    mx.clear_cache()

    diff = max_abs_diff(patched_first_logits, unpatched_first_logits)
    assert diff == 0.0, (
        "Gemma 4 text-only generation logits mismatch between patched and "
        f"unpatched mlx-lm (max diff {diff:.6f})."
    )


def test_gemma4_image_prompt_unified_arch_top5_matches_vlm():
    """Image+text Gemma 4 generation should stay close to native mlx-vlm."""
    model_path = get_real_model_path(GEMMA4_MODEL_NAME)
    image_b64 = read_image_b64(
        Path(__file__).parent.parent / "demo-data" / "toucan.jpeg"
    )
    prompt = build_gemma4_prompt(model_path, "What is this?", image_b64=image_b64)
    prefill_step_size = 2048

    model_kit = load_model(
        model_path=model_path,
        max_seq_nums=1,
        prefill_step_size=prefill_step_size,
    )
    prompt_tokens = model_kit.tokenize(prompt)
    input_tokens, input_embeddings = model_kit.process_prompt(
        prompt_tokens,
        images_b64=[image_b64],
        prompt_progress_reporter=DefaultPromptProgressReporter(),
        generate_args={},
        max_image_size=(1024, 1024),
    )
    assert input_embeddings is not None
    unified_first_logits = first_mlx_lm_generation_logits(
        model_kit.model,
        input_tokens,
        input_embeddings=input_embeddings,
        prefill_step_size=prefill_step_size,
    )
    model_kit.shutdown()
    del model_kit
    mx.clear_cache()

    vlm_model = load_vlm(model_path)
    vlm_processor = load_vlm_processor(model_path)
    vlm_inputs = prepare_inputs(
        processor=vlm_processor,
        images=convert_to_pil([image_b64]),
        prompts=prompt,
        image_token_index=resolve_image_token_index(vlm_model.config),
        resize_shape=None,
    )
    native_input_ids = vlm_inputs["input_ids"]
    native_attention_mask = vlm_inputs["attention_mask"]
    native_pixel_values = vlm_inputs["pixel_values"]

    assert input_tokens.tolist() == native_input_ids[0].tolist()

    vlm_first_logits = first_vlm_generation_logits(
        vlm_model,
        input_ids=native_input_ids,
        pixel_values=native_pixel_values,
        attention_mask=native_attention_mask,
        prefill_step_size=prefill_step_size,
    )
    tokenizer = (
        vlm_processor.tokenizer
        if hasattr(vlm_processor, "tokenizer")
        else vlm_processor
    )
    del vlm_model
    mx.clear_cache()

    unified_top5_ids = topk_token_ids(unified_first_logits, GEMMA4_IMAGE_TOPK)
    vlm_top5_ids = topk_token_ids(vlm_first_logits, GEMMA4_IMAGE_TOPK)
    unified_logits = gather_values(unified_first_logits, unified_top5_ids)
    vlm_logits = gather_values(vlm_first_logits, unified_top5_ids)
    unified_probabilities = softmax_probabilities(unified_first_logits)
    vlm_probabilities = softmax_probabilities(vlm_first_logits)
    unified_top5_probabilities = gather_values(unified_probabilities, unified_top5_ids)
    vlm_top5_probabilities = gather_values(vlm_probabilities, unified_top5_ids)

    assert unified_top5_ids == vlm_top5_ids, (
        "Top-5 token IDs/order mismatch: "
        f"unified={format_token_values(unified_top5_ids, unified_top5_probabilities, tokenizer)} "
        f"vlm={format_token_values(vlm_top5_ids, gather_values(vlm_probabilities, vlm_top5_ids), tokenizer)}"
    )

    relative_diffs = relative_differences(
        unified_top5_probabilities,
        vlm_top5_probabilities,
        GEMMA4_IMAGE_TOPK_PROB_REF_FLOOR,
    )
    max_relative_diff = max(relative_diffs)

    assert max_relative_diff <= GEMMA4_IMAGE_TOPK_PROB_RTOL, (
        "Top-5 probabilities exceeded tolerance: "
        f"max relative diff {max_relative_diff:.6f}; "
        f"relative_diffs={relative_diffs}; "
        f"unified_logits={format_token_values(unified_top5_ids, unified_logits, tokenizer)} "
        f"vlm_logits={format_token_values(unified_top5_ids, vlm_logits, tokenizer)} "
        f"unified_probabilities={format_token_values(unified_top5_ids, unified_top5_probabilities, tokenizer)} "
        f"vlm_probabilities={format_token_values(unified_top5_ids, vlm_top5_probabilities, tokenizer)}"
    )
