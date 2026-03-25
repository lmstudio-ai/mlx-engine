"""
Tests for monkey-patched model classes (mlx_engine.model_kit.patches).

Follows the upstream mlx-lm/tests/test_models.py pattern: construct models
from small configs with random weights, test through the public interface.
"""

import unittest

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


class TestPatchedQwen3_5(unittest.TestCase):
    """Tests for the Qwen3.5 MRoPE patch."""

    @classmethod
    def setUpClass(cls):
        from mlx_engine.model_kit.patches.qwen3_5 import apply_patches

        apply_patches()

        from mlx_lm.models.qwen3_5 import Model, ModelArgs

        cls.Model = Model
        cls.ModelArgs = ModelArgs

    def _make_model(self):
        args = self.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5",
                "text_config": {
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
                },
            }
        )
        return self.Model(args)

    def test_prefill_decode_consistency_with_mrope(self):
        """Full-sequence prefill and incremental prefill+decode must produce
        the same last-token logits when MRoPE state is active.

        A correct autoregressive model satisfies:
            model(tokens[0..N], embeddings[0..N])[-1]
                == model(tokens[0..N-1], embeddings[0..N-1]) ; model(tokens[N])

        When MRoPE state is cleared on the decode step, the decode token uses
        fallback sequential positions instead of the MRoPE-adjusted positions,
        breaking this invariant.
        """
        model = self._make_model()
        text_model = model.language_model.model

        tokens = mx.array([[0, 1, 2, 3]])
        embeddings = text_model.embed_tokens(tokens)

        # Non-degenerate 3D positions: height dim offset by 50, simulating
        # image spatial positions that differ across MRoPE dimensions.
        position_ids = mx.array([[[0, 1, 2, 3]], [[50, 51, 52, 53]], [[0, 1, 2, 3]]])
        rope_deltas = mx.array(-2)

        # Full prefill: all 4 tokens at once with cache
        cache_full = make_prompt_cache(model)
        text_model._position_ids = position_ids
        text_model._rope_deltas = rope_deltas
        full_output = model(tokens, cache=cache_full, input_embeddings=embeddings)
        mx.eval(full_output)
        full_last_logits = full_output[0, -1, :]

        # Incremental: prefill 3 tokens, then decode token 4
        cache_incr = make_prompt_cache(model)
        text_model._position_ids = position_ids
        text_model._rope_deltas = rope_deltas
        model(tokens[:, :-1], cache=cache_incr, input_embeddings=embeddings[:, :-1])
        mx.eval([c.state for c in cache_incr])

        decode_output = model(tokens[:, -1:], cache=cache_incr)
        mx.eval(decode_output)
        decode_logits = decode_output[0, -1, :]

        self.assertTrue(
            mx.allclose(full_last_logits, decode_logits, atol=1e-4).item(),
            f"Prefill-decode logit mismatch (max diff "
            f"{mx.max(mx.abs(full_last_logits - decode_logits)).item():.6f}). "
            f"MRoPE state was likely cleared during the decode step.",
        )


if __name__ == "__main__":
    unittest.main()
