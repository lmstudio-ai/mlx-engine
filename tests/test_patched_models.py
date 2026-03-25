"""
Tests for monkey-patched model classes (mlx_engine.model_kit.patches).

Follows the upstream mlx-lm/tests/test_models.py pattern: construct models
from small configs with random weights, test through the public interface.
"""

import pytest

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlx_engine.model_kit.patches.qwen3_5 import apply_patches

apply_patches()

from mlx_lm.models.qwen3_5 import Model, ModelArgs  # noqa: E402

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


def make_model():
    args = ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": QWEN3_5_TEXT_CONFIG,
        }
    )
    return Model(args)


@pytest.mark.parametrize("use_mrope", [False, True], ids=["text_only", "mrope"])
def test_qwen3_5_prefill_decode_consistency(use_mrope):
    """Full-sequence prefill and incremental prefill+decode must produce
    the same last-token logits.

    A correct autoregressive model satisfies:
        model(all_tokens)[-1] == model(tokens[:-1]); model(tokens[-1])

    Parameterized over:
      - text_only: standard RoPE path (no input_embeddings, no MRoPE state)
      - mrope: MRoPE path with non-degenerate 3D positions and input_embeddings,
               simulating a vision request
    """
    model = make_model()
    text_model = model.language_model.model
    tokens = mx.array([[0, 1, 2, 3]])

    if use_mrope:
        embeddings = text_model.embed_tokens(tokens)
        # 3D positions simulating a vision prompt where image tokens create
        # different spatial positions across temporal/height/width dims.
        # rope_deltas and position_ids must be consistent: the last token's
        # position (1) must equal cache_offset (3) + rope_deltas (-2).
        position_ids = mx.array(
            [
                [[0, 1, 0, 1]],  # temporal
                [[0, 0, 1, 1]],  # height — differs from dim 0 during prefill
                [[0, 1, 1, 1]],  # width  — differs from both during prefill
            ]
        )
        rope_deltas = mx.array(-2)
    else:
        embeddings = None
        position_ids = None
        rope_deltas = None

    # Full prefill: all tokens at once with cache
    cache_full = make_prompt_cache(model)
    text_model.position_ids = position_ids
    text_model.rope_deltas = rope_deltas
    full_output = model(tokens, cache=cache_full, input_embeddings=embeddings)
    mx.eval(full_output)
    full_last_logits = full_output[0, -1, :]

    # Incremental: prefill N-1 tokens, then decode last token
    cache_incr = make_prompt_cache(model)
    text_model.position_ids = position_ids
    text_model.rope_deltas = rope_deltas
    model(
        tokens[:, :-1],
        cache=cache_incr,
        input_embeddings=embeddings[:, :-1] if embeddings is not None else None,
    )
    mx.eval([c.state for c in cache_incr])

    decode_output = model(tokens[:, -1:], cache=cache_incr)
    mx.eval(decode_output)
    decode_logits = decode_output[0, -1, :]

    max_diff = mx.max(mx.abs(full_last_logits - decode_logits)).item()
    assert mx.allclose(full_last_logits, decode_logits, atol=1e-4).item(), (
        f"Prefill-decode logit mismatch (max diff {max_diff:.6f})."
    )
