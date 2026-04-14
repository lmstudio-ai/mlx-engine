"""Tests for the Gemma 4 monkey patch."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import mlx.core as mx
import mlx_lm.models.gemma4_text as gemma4_text_module

from mlx_engine.generate import load_model
from mlx_engine.model_kit.patches.gemma4 import (
    OriginalGemma4TextModel,
    PatchedGemma4TextModel,
)
from mlx_engine.utils.image_utils import convert_to_pil
from mlx_engine.utils.prompt_progress_reporter import DefaultPromptProgressReporter
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

GEMMA4_MODEL_NAME = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
GEMMA4_IMAGE_PROMPT_EMBEDDINGS_ATOL = 0.05


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


def test_patched_gemma4_slices_stored_prompt_ids_before_forward():
    model = PatchedGemma4TextModel.__new__(PatchedGemma4TextModel)
    model.prompt_per_layer_input_ids = mx.array([[10, 11, 12, 13, 14]], dtype=mx.int32)
    expected_input_ids = mx.array([[12, 13]], dtype=mx.int32)
    input_embeddings = mx.zeros((1, 2, 8), dtype=mx.float32)

    with (
        patch.object(
            PatchedGemma4TextModel,
            "_get_per_layer_inputs",
            autospec=True,
            return_value=expected_input_ids,
        ) as get_per_layer_inputs,
        patch.object(
            OriginalGemma4TextModel,
            "__call__",
            autospec=True,
            return_value="forwarded",
        ) as super_call,
    ):
        result = model(
            inputs=mx.array([[0, 1]], dtype=mx.int32),
            cache=[SimpleNamespace(offset=2)],
            input_embeddings=input_embeddings,
        )

    assert result == "forwarded"
    assert mx.array_equal(get_per_layer_inputs.call_args.args[1], expected_input_ids)
    assert super_call.call_args.kwargs["per_layer_inputs"] is expected_input_ids


@pytest.mark.heavy
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


@pytest.mark.heavy
def test_gemma4_image_prompt_unified_arch_prompt_inputs_match_vlm():
    """Image+text Gemma 4 prompt inputs should match native mlx-vlm before LM."""
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
    patched_text_model = model_kit.model.language_model.model
    prompt_per_layer_input_ids = patched_text_model.prompt_per_layer_input_ids
    assert prompt_per_layer_input_ids is not None
    unified_prompt_embeddings = (
        input_embeddings[None].astype(mx.float32) * patched_text_model.embed_scale
    )
    mx.eval(unified_prompt_embeddings)
    mx.eval(prompt_per_layer_input_ids)
    model_kit.shutdown()
    del model_kit
    mx.clear_cache()

    vlm_model = load_vlm(model_path)
    image_token_index = resolve_image_token_index(vlm_model.config)
    vlm_processor = load_vlm_processor(model_path)
    vlm_inputs = prepare_inputs(
        processor=vlm_processor,
        images=convert_to_pil([image_b64]),
        prompts=prompt,
        image_token_index=image_token_index,
        resize_shape=None,
    )
    native_input_ids = vlm_inputs["input_ids"]
    native_attention_mask = vlm_inputs["attention_mask"]
    native_pixel_values = vlm_inputs["pixel_values"]
    native_embedding_output = vlm_model.get_input_embeddings(
        input_ids=native_input_ids,
        pixel_values=native_pixel_values,
        mask=native_attention_mask,
    )
    native_prompt_embeddings = native_embedding_output.inputs_embeds.astype(mx.float32)
    expected_prompt_per_layer_input_ids = mx.where(
        native_input_ids == image_token_index, 0, native_input_ids
    )
    mx.eval(expected_prompt_per_layer_input_ids)

    assert input_tokens.tolist() == native_input_ids[0].tolist()
    mx.eval(native_prompt_embeddings)
    del vlm_model
    mx.clear_cache()

    prompt_embedding_diff = max_abs_diff(
        unified_prompt_embeddings, native_prompt_embeddings
    )
    assert mx.allclose(
        unified_prompt_embeddings,
        native_prompt_embeddings,
        atol=GEMMA4_IMAGE_PROMPT_EMBEDDINGS_ATOL,
        rtol=0.0,
    ).item(), (
        "Prompt embeddings mismatch against native mlx-vlm before LM decode "
        f"(max diff {prompt_embedding_diff:.6f})."
    )

    assert mx.all(
        prompt_per_layer_input_ids == expected_prompt_per_layer_input_ids
    ).item(), "Stored prompt_per_layer_input_ids mismatch native masked prompt tokens."
