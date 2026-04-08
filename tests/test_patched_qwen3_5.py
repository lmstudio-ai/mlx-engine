"""Tests for the Qwen3.5 monkey patches."""

import pytest

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache
from mlx_lm.models.qwen3_5 import Model, ModelArgs
import mlx_lm.models.qwen3_5 as qwen3_5_module

from mlx_engine.model_kit.vision_add_ons.qwen3_5 import _compute_image_mrope_state
from mlx_engine.model_kit.patches.qwen3_5 import (
    OriginalDecoderLayer,
    OriginalQwen3_5TextModel,
)

from tests.patched_model_test_utils import (
    assert_restorable_binding,
    get_real_model_path,
    load_unpatched_mlx_lm,
    load_patched_mlx_lm,
    load_vlm,
    max_abs_diff,
)

REAL_MODEL_CASES = [
    pytest.param("lmstudio-community/Qwen3.5-2B-MLX-4bit", id="dense"),
    pytest.param(
        "lmstudio-community/Qwen3.5-35B-A3B-MLX-4bit",
        marks=pytest.mark.heavy,
        id="moe",
    ),
]

QWEN3_5_TEXT_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 1000,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 3,
    "rms_norm_eps": 1e-5,
    "head_dim": 64,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 1000,
}

QWEN3_5_MROPE_CHUNK_CASES = [
    pytest.param(
        {
            "tokens": [0, 1, 2, 3, 4, 5],
            "position_ids": [
                [[0, 1, 1, 1, 1, 3]],
                [[0, 1, 1, 2, 2, 3]],
                [[0, 1, 2, 1, 2, 3]],
            ],
            "rope_deltas": -2,
            "failure_label": "Chunked MRoPE prefill mismatch",
        },
        id="crosses_chunk_boundary",
    ),
    pytest.param(
        {
            "tokens": [0, 1, 2, 3, 4, 5],
            "position_ids": [
                [[0, 1, 2, 2, 4, 5]],
                [[0, 1, 2, 2, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
            ],
            "rope_deltas": 0,
            "failure_label": "Later-chunk MRoPE prefill mismatch",
        },
        id="image_span_in_later_chunk",
    ),
]


def make_model(**text_config_overrides):
    args = ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                **QWEN3_5_TEXT_CONFIG,
                **text_config_overrides,
            },
        }
    )
    return Model(args)


def make_batched_prompt_cache(model, left_padding):
    cache = model.make_cache()
    for index, layer_cache in enumerate(cache):
        if isinstance(layer_cache, ArraysCache):
            layer_cache.left_padding = mx.array(left_padding)
        elif type(layer_cache) is KVCache:
            cache[index] = BatchKVCache(left_padding)
        else:
            raise AssertionError(f"Unexpected cache type: {type(layer_cache)!r}")
    return cache


def load_unpatched_qwen_mlx_lm(model_path):
    assert_restorable_binding(
        OriginalDecoderLayer,
        qwen3_5_module.DecoderLayer,
        "qwen3.5 DecoderLayer",
    )
    assert_restorable_binding(
        OriginalQwen3_5TextModel,
        qwen3_5_module.Qwen3_5TextModel,
        "qwen3.5 Qwen3_5TextModel",
    )
    return load_unpatched_mlx_lm(
        model_path,
        module=qwen3_5_module,
        replacements={
            "DecoderLayer": OriginalDecoderLayer,
            "Qwen3_5TextModel": OriginalQwen3_5TextModel,
        },
    )


def _first_generate_step_logprobs(
    model,
    tokens: mx.array,
    *,
    input_embeddings: mx.array | None = None,
    position_ids: mx.array | None = None,
    rope_deltas: mx.array | None = None,
    prefill_step_size: int,
) -> mx.array:
    text_model = model.language_model.model
    text_model.position_ids = position_ids
    text_model.rope_deltas = rope_deltas
    step = generate_step(
        tokens,
        model,
        max_tokens=1,
        prefill_step_size=prefill_step_size,
        input_embeddings=input_embeddings,
    )
    _, logprobs = next(step)
    step.close()
    mx.eval(logprobs)
    return logprobs


@pytest.mark.parametrize("use_mrope", [False, True], ids=["text_only", "mrope"])
def test_qwen3_5_prefill_decode_consistency(use_mrope):
    """Full-sequence prefill and incremental prefill+decode must agree."""
    model = make_model()
    text_model = model.language_model.model
    tokens = mx.array([[0, 1, 2, 3]])

    if use_mrope:
        embeddings = text_model.embed_tokens(tokens)
        position_ids = mx.array(
            [
                [[0, 1, 0, 1]],
                [[0, 0, 1, 1]],
                [[0, 1, 1, 1]],
            ]
        )
        rope_deltas = mx.array(-2)
    else:
        embeddings = None
        position_ids = None
        rope_deltas = None

    cache_full = make_prompt_cache(model)
    text_model.position_ids = position_ids
    text_model.rope_deltas = rope_deltas
    full_output = model(tokens, cache=cache_full, input_embeddings=embeddings)
    mx.eval(full_output)
    full_last_logits = full_output[0, -1, :]

    cache_incr = make_prompt_cache(model)
    text_model.position_ids = position_ids
    text_model.rope_deltas = rope_deltas
    model(
        tokens[:, :-1],
        cache=cache_incr,
        input_embeddings=embeddings[:, :-1] if embeddings is not None else None,
    )
    mx.eval([cache.state for cache in cache_incr])

    decode_output = model(tokens[:, -1:], cache=cache_incr)
    mx.eval(decode_output)
    decode_logits = decode_output[0, -1, :]

    diff = max_abs_diff(full_last_logits, decode_logits)
    assert mx.allclose(full_last_logits, decode_logits, atol=1e-4).item(), (
        f"Prefill-decode logit mismatch (max diff {diff:.6f})."
    )


@pytest.mark.parametrize("case", QWEN3_5_MROPE_CHUNK_CASES)
def test_qwen3_5_mrope_chunked_prefill_matches_unchunked(case):
    """Chunked MRoPE prefill must match unchunked prefill."""
    mx.random.seed(0)

    model = make_model(
        num_hidden_layers=2,
        full_attention_interval=2,
    )
    text_model = model.language_model.model
    tokens = mx.array(case["tokens"])
    embeddings = text_model.embed_tokens(tokens)
    position_ids = mx.array(case["position_ids"])
    rope_deltas = mx.array(case["rope_deltas"])

    reference_logprobs = _first_generate_step_logprobs(
        model,
        tokens,
        input_embeddings=embeddings,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        prefill_step_size=16,
    )
    chunked_logprobs = _first_generate_step_logprobs(
        model,
        tokens,
        input_embeddings=embeddings,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        prefill_step_size=2,
    )

    diff = max_abs_diff(reference_logprobs, chunked_logprobs)
    assert mx.allclose(reference_logprobs, chunked_logprobs, atol=1e-4).item(), (
        f"{case['failure_label']} (max diff {diff:.6f})."
    )


def test_qwen3_5_text_only_uncached_matches_prompt_cache():
    """Direct uncached text-only forwards should match the cached prefill path."""
    model = make_model()
    tokens = mx.array([[0, 1, 2, 3]])

    reference_cache = make_prompt_cache(model)
    reference_logits = model(tokens, cache=reference_cache)
    mx.eval(reference_logits)

    uncached_logits = model(tokens, cache=None)
    mx.eval(uncached_logits)

    diff = max_abs_diff(reference_logits, uncached_logits)
    assert mx.allclose(reference_logits, uncached_logits, atol=1e-4).item(), (
        f"Text-only uncached logits mismatch (max diff {diff:.6f})."
    )


def test_qwen3_5_text_only_batch_cache_matches_prompt_cache():
    """Text-only forwards should handle BatchKVCache vector offsets."""
    model = make_model()
    text_model = model.language_model.model
    tokens = mx.array([[0, 1, 2, 3]])

    reference_cache = make_prompt_cache(model)
    reference_logits = model(tokens, cache=reference_cache)
    mx.eval(reference_logits)

    batch_cache = make_batched_prompt_cache(model, [0])
    assert isinstance(batch_cache[text_model.fa_idx], BatchKVCache)
    assert batch_cache[text_model.fa_idx].offset.ndim == 1

    batch_logits = model(tokens, cache=batch_cache)
    mx.eval(batch_logits)

    diff = max_abs_diff(reference_logits, batch_logits)
    assert mx.allclose(reference_logits, batch_logits, atol=1e-4).item(), (
        f"Text-only batch-cache logits mismatch (max diff {diff:.6f})."
    )


@pytest.mark.parametrize("model_name", REAL_MODEL_CASES)
def test_qwen3_5_text_only_patched_matches_unpatched(model_name):
    """The Qwen3.5 patch must be a no-op for text-only inference."""
    model_path = get_real_model_path(model_name)
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    patched_model, _ = load_patched_mlx_lm(model_path)
    patched_logits = patched_model(tokens)
    mx.eval(patched_logits)
    patched_logits = mx.array(patched_logits)
    del patched_model
    mx.clear_cache()

    unpatched_model, _ = load_unpatched_qwen_mlx_lm(model_path)
    unpatched_logits = unpatched_model(tokens)
    mx.eval(unpatched_logits)
    unpatched_logits = mx.array(unpatched_logits)
    del unpatched_model
    mx.clear_cache()

    diff = max_abs_diff(patched_logits, unpatched_logits)
    assert diff == 0.0, (
        f"{model_name}: patched vs unpatched mlx-lm logits mismatch "
        f"(max diff {diff:.6f})."
    )


@pytest.mark.parametrize("model_name", REAL_MODEL_CASES)
def test_qwen3_5_image_prompt_patched_matches_vlm(model_name):
    """The patched Qwen3.5 image path must match native mlx-vlm."""
    model_path = get_real_model_path(model_name)
    vlm_model = load_vlm(model_path)
    config = vlm_model.config

    image_grid_thw = mx.array([[1, 4, 4]])
    tokens_list = [
        0,
        1,
        config.vision_start_token_id,
        config.image_token_id,
        config.image_token_id,
        config.image_token_id,
        config.image_token_id,
        2,
        3,
    ]
    tokens = mx.array([tokens_list])

    vlm_position_ids, vlm_rope_deltas = vlm_model.language_model.get_rope_index(
        tokens, image_grid_thw=image_grid_thw
    )
    mx.eval(vlm_position_ids, vlm_rope_deltas)

    addon_position_ids, addon_rope_deltas = _compute_image_mrope_state(
        mx.array(tokens_list), image_grid_thw, config
    )
    mx.eval(addon_position_ids, addon_rope_deltas)

    assert mx.array_equal(addon_position_ids, vlm_position_ids).item(), (
        f"{model_name}: position IDs mismatch: addon={addon_position_ids.tolist()}, "
        f"vlm={vlm_position_ids.tolist()}"
    )
    assert mx.array_equal(addon_rope_deltas, vlm_rope_deltas).item(), (
        f"{model_name}: rope deltas mismatch: addon={addon_rope_deltas.item()}, "
        f"vlm={vlm_rope_deltas.item()}"
    )

    vlm_model.language_model._position_ids = vlm_position_ids
    vlm_model.language_model._rope_deltas = vlm_rope_deltas
    vlm_logits = vlm_model.language_model(
        tokens, cache=None, position_ids=vlm_position_ids
    ).logits
    mx.eval(vlm_logits)
    vlm_logits = mx.array(vlm_logits)
    del vlm_model
    mx.clear_cache()

    patched_model, _ = load_patched_mlx_lm(model_path)
    patched_text_model = patched_model.language_model.model
    patched_text_model.position_ids = addon_position_ids
    patched_text_model.rope_deltas = addon_rope_deltas
    patched_logits = patched_model(tokens)
    mx.eval(patched_logits)
    patched_logits = mx.array(patched_logits)
    del patched_model
    mx.clear_cache()

    diff = max_abs_diff(patched_logits, vlm_logits)
    assert diff == 0.0, (
        f"{model_name}: image prompt logits mismatch against mlx-vlm "
        f"(max diff {diff:.6f})."
    )
