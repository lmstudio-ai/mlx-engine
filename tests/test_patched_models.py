"""
Tests for monkey-patched model classes (mlx_engine.model_kit.patches).

Follows the upstream mlx-lm/tests/test_models.py pattern: construct models
from small configs with random weights, test through the public interface.

Also includes cross-engine tests that load a real model from disk and compare
patched mlx-lm logits against unpatched mlx-lm and native mlx-vlm.
"""

import pytest
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache

# Save original (unpatched) classes before applying patches
from mlx_lm.models.qwen3_5 import (
    DecoderLayer as _OrigDecoderLayer,
    Qwen3_5TextModel as _OrigTextModel,
)

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


def make_batched_prompt_cache(model, left_padding):
    cache = model.make_cache()
    for i, layer_cache in enumerate(cache):
        if isinstance(layer_cache, ArraysCache):
            layer_cache.left_padding = mx.array(left_padding)
        elif type(layer_cache) is KVCache:
            cache[i] = BatchKVCache(left_padding)
        else:
            raise AssertionError(f"Unexpected cache type: {type(layer_cache)!r}")
    return cache


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


def test_qwen3_5_text_only_uncached_matches_prompt_cache():
    """Direct uncached text-only forwards should match the cached prefill path."""
    model = make_model()
    tokens = mx.array([[0, 1, 2, 3]])

    reference_cache = make_prompt_cache(model)
    reference_logits = model(tokens, cache=reference_cache)
    mx.eval(reference_logits)

    uncached_logits = model(tokens, cache=None)
    mx.eval(uncached_logits)

    max_diff = mx.max(mx.abs(reference_logits - uncached_logits)).item()
    assert mx.allclose(reference_logits, uncached_logits, atol=1e-4).item(), (
        f"Text-only uncached logits mismatch (max diff {max_diff:.6f})."
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

    max_diff = mx.max(mx.abs(reference_logits - batch_logits)).item()
    assert mx.allclose(reference_logits, batch_logits, atol=1e-4).item(), (
        f"Text-only batch-cache logits mismatch (max diff {max_diff:.6f})."
    )


# ---------------------------------------------------------------------------
# Cross-engine tests: compare patched mlx-lm against unpatched and mlx-vlm
# using a real model loaded from disk.
# ---------------------------------------------------------------------------

REAL_MODEL_NAME = "lmstudio-community/Qwen3.5-2B-MLX-4bit"


def _get_real_model_path() -> Path:
    from tests.shared import model_getter

    return model_getter(REAL_MODEL_NAME)


def _load_patched_mlx_lm(model_path: Path):
    """Load model using mlx-lm with patches already applied."""
    import mlx_lm.utils

    model, tokenizer = mlx_lm.utils.load(model_path)
    return model, tokenizer


def _load_unpatched_mlx_lm(model_path: Path):
    """Load model using mlx-lm with original (unpatched) classes temporarily restored."""
    import mlx_lm.models.qwen3_5 as mod
    import mlx_lm.utils

    patched_dl = mod.DecoderLayer
    patched_tm = mod.Qwen3_5TextModel
    mod.DecoderLayer = _OrigDecoderLayer
    mod.Qwen3_5TextModel = _OrigTextModel
    try:
        model, tokenizer = mlx_lm.utils.load(model_path)
    finally:
        mod.DecoderLayer = patched_dl
        mod.Qwen3_5TextModel = patched_tm
    return model, tokenizer


def _load_vlm(model_path: Path):
    """Load model using mlx-vlm's native loader."""
    from mlx_vlm.utils import load_model as vlm_load_model

    return vlm_load_model(model_path)


def test_qwen3_5_text_only_patched_matches_unpatched_and_vlm():
    """Text-only logits from the patched mlx-lm model must match both the
    unpatched mlx-lm model and the native mlx-vlm LanguageModel.

    This validates that the MRoPE patch is a no-op for text-only inference:
    the patched model's sequential 3D positions collapse to standard RoPE,
    producing identical logits to the unpatched original.

    Models are loaded and unloaded sequentially to limit memory usage.
    """
    model_path = _get_real_model_path()
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]])

    # --- Patched mlx-lm ---
    patched_model, _ = _load_patched_mlx_lm(model_path)
    patched_logits = patched_model(tokens)
    mx.eval(patched_logits)
    # Detach from model graph before unloading
    patched_logits = mx.array(patched_logits)
    del patched_model
    mx.clear_cache()

    # --- Unpatched mlx-lm ---
    unpatched_model, _ = _load_unpatched_mlx_lm(model_path)
    unpatched_logits = unpatched_model(tokens)
    mx.eval(unpatched_logits)
    unpatched_logits = mx.array(unpatched_logits)
    del unpatched_model
    mx.clear_cache()

    # --- mlx-vlm ---
    vlm_model = _load_vlm(model_path)
    # Pass explicit sequential position_ids to skip get_rope_index (which
    # needs a full ModelConfig). Shape: (3, batch, seq) — all 3 dims identical
    # for text-only, equivalent to standard RoPE.
    seq_len = tokens.shape[1]
    position_ids = mx.broadcast_to(
        mx.arange(seq_len).reshape(1, 1, seq_len),
        (3, 1, seq_len),
    )
    vlm_logits = vlm_model.language_model(
        tokens, cache=None, position_ids=position_ids
    ).logits
    mx.eval(vlm_logits)
    vlm_logits = mx.array(vlm_logits)
    del vlm_model
    mx.clear_cache()

    # --- Compare: run all comparisons before failing ---
    vlm_atol = 0.5
    diff_patched_unpatched = mx.max(mx.abs(patched_logits - unpatched_logits)).item()
    diff_patched_vlm = mx.max(mx.abs(patched_logits - vlm_logits)).item()
    diff_unpatched_vlm = mx.max(mx.abs(unpatched_logits - vlm_logits)).item()

    failures = []
    if diff_patched_unpatched != 0.0:
        failures.append(
            f"Patched vs unpatched mlx-lm: max diff {diff_patched_unpatched:.6f}"
        )
    if not mx.allclose(patched_logits, vlm_logits, atol=vlm_atol).item():
        failures.append(f"Patched mlx-lm vs mlx-vlm: max diff {diff_patched_vlm:.6f}")
    if not mx.allclose(unpatched_logits, vlm_logits, atol=vlm_atol).item():
        failures.append(
            f"Unpatched mlx-lm vs mlx-vlm: max diff {diff_unpatched_vlm:.6f}"
        )

    summary = (
        f"\n  patched vs unpatched: {diff_patched_unpatched:.6f}"
        f"\n  patched vs vlm:      {diff_patched_vlm:.6f}"
        f"\n  unpatched vs vlm:    {diff_unpatched_vlm:.6f}"
    )
    assert len(failures) == 0, (
        f"Logit mismatch (atol={vlm_atol}):{summary}\nFailures: {'; '.join(failures)}"
    )


def test_qwen3_5_image_prompt_patched_matches_vlm():
    """Image-prompt logits from the patched mlx-lm model must match the native
    mlx-vlm LanguageModel.

    This validates two things:
    1. The vision add-on's _compute_image_mrope_state produces the same 3D
       position IDs as mlx-vlm's LanguageModel.get_rope_index.
    2. Given those positions, the patched model's forward pass produces the
       same logits as mlx-vlm (within bfloat16 tolerance from different
       GatedDeltaNet implementations).

    Uses a synthetic token sequence with image token runs — no actual image
    pixels are processed. The test focuses on position computation and
    position threading through the model.

    Models are loaded and unloaded sequentially to limit memory usage.
    """
    from mlx_engine.model_kit.vision_add_ons.qwen3_5 import _compute_image_mrope_state

    model_path = _get_real_model_path()

    # --- Load vlm model first to get config and compute reference positions ---
    vlm_model = _load_vlm(model_path)
    config = vlm_model.config

    # Construct a synthetic token sequence with an image span.
    # Layout: [text, text, vision_start, image*4, text, text]
    # With spatial_merge_size=2 and grid_thw=[1,4,4]:
    #   llm_grid = [1, 2, 2] → 4 image tokens
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

    # Compute positions via mlx-vlm's get_rope_index
    vlm_position_ids, vlm_rope_deltas = vlm_model.language_model.get_rope_index(
        tokens, image_grid_thw=image_grid_thw
    )
    mx.eval(vlm_position_ids, vlm_rope_deltas)

    # Compute positions via the vision add-on's method
    addon_position_ids, addon_rope_deltas = _compute_image_mrope_state(
        mx.array(tokens_list), image_grid_thw, config
    )
    mx.eval(addon_position_ids, addon_rope_deltas)

    # --- Assert position computation matches ---
    position_ids_match = mx.array_equal(addon_position_ids, vlm_position_ids).item()
    rope_deltas_match = mx.array_equal(addon_rope_deltas, vlm_rope_deltas).item()

    # --- Forward pass: vlm ---
    vlm_model.language_model._position_ids = vlm_position_ids
    vlm_model.language_model._rope_deltas = vlm_rope_deltas
    vlm_logits = vlm_model.language_model(
        tokens, cache=None, position_ids=vlm_position_ids
    ).logits
    mx.eval(vlm_logits)
    vlm_logits = mx.array(vlm_logits)
    del vlm_model
    mx.clear_cache()

    # --- Forward pass: patched mlx-lm ---
    patched_model, _ = _load_patched_mlx_lm(model_path)
    patched_text_model = patched_model.language_model.model
    patched_text_model.position_ids = addon_position_ids
    patched_text_model.rope_deltas = addon_rope_deltas
    patched_logits = patched_model(tokens)
    mx.eval(patched_logits)
    patched_logits = mx.array(patched_logits)
    del patched_model
    mx.clear_cache()

    # --- Compare: run all checks before failing ---
    diff_logits = mx.max(mx.abs(patched_logits - vlm_logits)).item()

    failures = []
    if not position_ids_match:
        failures.append(
            f"Position IDs mismatch: addon={addon_position_ids.tolist()}, "
            f"vlm={vlm_position_ids.tolist()}"
        )
    if not rope_deltas_match:
        failures.append(
            f"Rope deltas mismatch: addon={addon_rope_deltas.item()}, "
            f"vlm={vlm_rope_deltas.item()}"
        )
    if diff_logits != 0.0:
        failures.append(f"Logit mismatch: max diff {diff_logits:.6f}")

    summary = (
        f"\n  position_ids match:  {position_ids_match}"
        f"\n  rope_deltas match:   {rope_deltas_match}"
        f"\n  logit max diff:      {diff_logits:.6f}"
    )
    assert len(failures) == 0, (
        f"Image prompt mismatch (expected zero diff):{summary}\nFailures: {'; '.join(failures)}"
    )
