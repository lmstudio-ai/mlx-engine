import mlx.core as mx
from mlx_lm.generate import generate_step

from mlx_engine.model_kit.patches.gemma4 import (
    OriginalGemma4TextModel as _OrigTextModel,
    PatchedGemma4TextModel,
    apply_patches,
)

apply_patches()

from mlx_lm.models.gemma4_text import Model, ModelArgs  # noqa: E402

GEMMA4_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "head_dim": 16,
    "global_head_dim": 16,
    "sliding_window": 8,
    "sliding_window_pattern": 1,
    "layer_types": ["full_attention", "full_attention"],
    "hidden_size_per_layer_input": 8,
    "vocab_size": 32,
    "vocab_size_per_layer_input": 32,
    "num_kv_shared_layers": 0,
}


def make_model(**text_config_overrides):
    args = ModelArgs.from_dict(
        {
            **GEMMA4_TEXT_CONFIG,
            **text_config_overrides,
        }
    )
    return Model(args)


def test_gemma4_text_only_patched_matches_unpatched():
    import mlx_lm.models.gemma4_text as mod

    patched_text_model = mod.Gemma4TextModel
    assert patched_text_model is PatchedGemma4TextModel
    assert _OrigTextModel is not PatchedGemma4TextModel

    mx.random.seed(0)
    model = make_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    patched_logits = model(tokens)
    mx.eval(patched_logits)
    patched_logits = mx.array(patched_logits)

    mod.Gemma4TextModel = _OrigTextModel
    try:
        mx.random.seed(0)
        unpatched_model = make_model()
    finally:
        mod.Gemma4TextModel = patched_text_model

    unpatched_logits = unpatched_model(tokens)
    mx.eval(unpatched_logits)
    max_diff = mx.max(mx.abs(patched_logits - unpatched_logits)).item()
    assert mx.allclose(patched_logits, unpatched_logits, atol=1e-4).item(), (
        f"Patched Gemma 4 text-only logits diverged from unpatched mlx-lm "
        f"(max diff {max_diff:.6f})."
    )


def test_gemma4_prompt_per_layer_inputs_chunked_prefill_matches_unchunked():
    mx.random.seed(0)

    model = make_model()
    text_model = model.model
    prompt = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
    prompt_embeddings = text_model.embed_tokens(prompt[None]).squeeze(0)
    prompt_per_layer_inputs = mx.full(
        (
            1,
            prompt.shape[0],
            text_model.config.num_hidden_layers,
            text_model.hidden_size_per_layer_input,
        ),
        3.0,
        dtype=prompt_embeddings.dtype,
    )

    def first_step_logprobs(prefill_step_size: int) -> mx.array:
        text_model.set_prompt_per_layer_inputs(prompt_per_layer_inputs)
        step = generate_step(
            prompt,
            model,
            max_tokens=1,
            sampler=lambda x: mx.argmax(x, axis=-1),
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
        )
        _, logprobs = next(step)
        step.close()
        mx.eval(logprobs)
        return logprobs

    reference_logprobs = first_step_logprobs(prefill_step_size=16)
    chunked_logprobs = first_step_logprobs(prefill_step_size=2)

    max_diff = mx.max(mx.abs(reference_logprobs - chunked_logprobs)).item()
    assert mx.allclose(reference_logprobs, chunked_logprobs, atol=1e-4).item(), (
        f"Chunked Gemma 4 prompt per-layer-input prefill mismatch "
        f"(max diff {max_diff:.6f})."
    )


def test_gemma4_prompt_state_matches_explicit_per_layer_inputs():
    mx.random.seed(0)

    model = make_model()
    text_model = model.model
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    prompt_embeddings = text_model.embed_tokens(prompt)
    prompt_per_layer_inputs = mx.full(
        (
            1,
            prompt.shape[1],
            text_model.config.num_hidden_layers,
            text_model.hidden_size_per_layer_input,
        ),
        2.0,
        dtype=prompt_embeddings.dtype,
    )

    explicit_logits = model(
        prompt,
        input_embeddings=prompt_embeddings,
        per_layer_inputs=prompt_per_layer_inputs,
    )
    text_model.set_prompt_per_layer_inputs(prompt_per_layer_inputs)
    stateful_logits = model(
        prompt,
        input_embeddings=prompt_embeddings,
    )
    mx.eval(explicit_logits, stateful_logits)

    max_diff = mx.max(mx.abs(explicit_logits - stateful_logits)).item()
    assert mx.allclose(explicit_logits, stateful_logits, atol=1e-4).item(), (
        f"Stored Gemma 4 prompt per-layer inputs diverged from explicit inputs "
        f"(max diff {max_diff:.6f})."
    )
