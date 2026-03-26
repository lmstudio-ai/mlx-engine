"""
Tests for monkey-patched model classes (mlx_engine.model_kit.patches).

Follows the upstream mlx-lm/tests/test_models.py pattern: construct models
from small configs with random weights, test through the public interface.
"""

import pytest

import mlx.core as mx
from mlx_lm.generate import generate_step
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


def test_qwen3_5_mrope_chunked_prefill_matches_unchunked():
    """Chunked prefill must match unchunked prefill even when a later prefill
    chunk is still inside a non-sequential MRoPE image span.

    The prompt is only 6 tokens long, but forcing prefill_step_size=2 reproduces
    the same chunking scenario as a 512-token prefill boundary in production:
      - chunk 1 processes text + first image token
      - chunk 2 processes image tokens while cache_offset > 0
    Later prompt chunks must continue using the stored 3D position_ids until the
    precomputed prompt positions are exhausted.
    """
    mx.random.seed(0)

    model = make_model(
        num_hidden_layers=2,
        full_attention_interval=2,
    )
    text_model = model.language_model.model

    tokens = mx.array([0, 1, 2, 3, 4, 5])
    embeddings = text_model.embed_tokens(tokens)

    # Synthetic prompt layout:
    #   token 0: text
    #   tokens 1-4: 2x2 image span with non-sequential 3D positions
    #   token 5: trailing text
    #
    # The final text token's position (3) equals cache_offset (5) + rope_deltas (-2),
    # so the reference unchunked path and decode path are consistent.
    position_ids = mx.array(
        [
            [[0, 1, 1, 1, 1, 3]],  # temporal
            [[0, 1, 1, 2, 2, 3]],  # height
            [[0, 1, 2, 1, 2, 3]],  # width
        ]
    )
    rope_deltas = mx.array(-2)

    def first_step_logprobs(prefill_step_size: int) -> mx.array:
        text_model.position_ids = position_ids
        text_model.rope_deltas = rope_deltas
        step = generate_step(
            tokens,
            model,
            max_tokens=1,
            prefill_step_size=prefill_step_size,
            input_embeddings=embeddings,
        )
        _, logprobs = next(step)
        step.close()
        mx.eval(logprobs)
        return logprobs

    # Single prefill chunk for tokens[:-1].
    reference_logprobs = first_step_logprobs(prefill_step_size=16)

    # Forces multiple prefill chunks; chunk 2 still lies inside the image span.
    chunked_logprobs = first_step_logprobs(prefill_step_size=2)

    max_diff = mx.max(mx.abs(reference_logprobs - chunked_logprobs)).item()
    assert mx.allclose(reference_logprobs, chunked_logprobs, atol=1e-4).item(), (
        f"Chunked MRoPE prefill mismatch (max diff {max_diff:.6f})."
    )


def test_qwen3_5_mrope_later_single_image_chunk_matches_unchunked():
    """Chunked prefill must match unchunked prefill even when the full image
    span is contained in a later prefill chunk.

    This isolates the more general bug: the implementation must keep using the
    stored multimodal prompt positions for any later chunk that still belongs to
    the original multimodal prompt. The image tokens do not cross a chunk
    boundary here; earlier text simply pushes them out of the first chunk.

    With prefill_step_size=2:
      - chunk 1 processes only leading text
      - chunk 2 processes the entire image span
      - chunk 3 processes trailing text
    """
    mx.random.seed(0)

    model = make_model(
        num_hidden_layers=2,
        full_attention_interval=2,
    )
    text_model = model.language_model.model

    tokens = mx.array([0, 1, 2, 3, 4, 5])
    embeddings = text_model.embed_tokens(tokens)

    # Synthetic prompt layout:
    #   tokens 0-1: leading text
    #   tokens 2-3: 1x2 image span with non-sequential 3D positions
    #   tokens 4-5: trailing text
    #
    # The image span is fully contained in chunk 2 when prefill_step_size=2.
    # Continuation after the prompt is sequential again, so rope_deltas is 0.
    position_ids = mx.array(
        [
            [[0, 1, 2, 2, 4, 5]],  # temporal
            [[0, 1, 2, 2, 4, 5]],  # height
            [[0, 1, 2, 3, 4, 5]],  # width
        ]
    )
    rope_deltas = mx.array(0)

    def first_step_logprobs(prefill_step_size: int) -> mx.array:
        text_model.position_ids = position_ids
        text_model.rope_deltas = rope_deltas
        step = generate_step(
            tokens,
            model,
            max_tokens=1,
            prefill_step_size=prefill_step_size,
            input_embeddings=embeddings,
        )
        _, logprobs = next(step)
        step.close()
        mx.eval(logprobs)
        return logprobs

    # Single prefill chunk for tokens[:-1].
    reference_logprobs = first_step_logprobs(prefill_step_size=16)

    # Chunk 2 contains the whole image span but is not the first prefill chunk.
    chunked_logprobs = first_step_logprobs(prefill_step_size=2)

    max_diff = mx.max(mx.abs(reference_logprobs - chunked_logprobs)).item()
    assert mx.allclose(reference_logprobs, chunked_logprobs, atol=1e-4).item(), (
        f"Later-chunk MRoPE prefill mismatch (max diff {max_diff:.6f})."
    )
