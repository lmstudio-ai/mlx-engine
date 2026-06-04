from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from mlx_engine.model_kit.patches.gemma4 import (
    patch_loaded_model,
    prepare_cached_suffix_prompt_kwargs,
)


class _Gemma4UnifiedTextModel:
    def _block_sequence_ids_for_mask(self, mm_token_type_ids):
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        prev = mx.concatenate(
            [mx.zeros_like(is_vision[:, :1]), is_vision[:, :-1]],
            axis=1,
        )
        starts = is_vision & ~prev
        group_ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
        return mx.where(is_vision, group_ids, mx.zeros_like(group_ids) - 1)

    def _apply_blockwise_bidirectional_overlay(self, base_mask, mm_token_type_ids):
        raise AssertionError("unpatched")


def test_gemma4_cached_suffix_prompt_kwargs_pad_visual_token_types_to_key_len():
    """Restored Gemma4 visual suffix masks need token types for cached keys."""
    prompt_kwargs = {
        "mm_token_type_ids": mx.array([[0, 1, 0, 2]], dtype=mx.int32),
        "unchanged": "value",
    }

    prepared = prepare_cached_suffix_prompt_kwargs(prompt_kwargs, key_len=7)

    assert prepared is not prompt_kwargs
    assert prepared["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 0, 2]]
    assert prepared["unchanged"] == "value"


def test_gemma4_cached_suffix_prompt_kwargs_keeps_text_only_token_types():
    prompt_kwargs = {
        "mm_token_type_ids": mx.array([[0, 0, 0, 0]], dtype=mx.int32),
    }

    prepared = prepare_cached_suffix_prompt_kwargs(prompt_kwargs, key_len=7)

    assert prepared is prompt_kwargs
    assert prepared["mm_token_type_ids"].tolist() == [[0, 0, 0, 0]]


def test_gemma4_suffix_visual_mask_patch_uses_query_rows_only():
    text_model = _Gemma4UnifiedTextModel()
    language_model = SimpleNamespace(
        model_type="gemma4_unified_text",
        model=text_model,
    )
    model = SimpleNamespace(language_model=language_model)

    patch_loaded_model(model)

    base_mask = create_causal_mask(4, offset=5)
    token_types = mx.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=mx.int32)
    patched = text_model._apply_blockwise_bidirectional_overlay(
        base_mask,
        token_types,
    )

    assert patched.shape == (1, 1, 4, 9)
    assert bool(patched[0, 0, 2, 8].item())
    assert not bool(base_mask[2, 8].item())

    short_token_types = mx.array([[0, 1]], dtype=mx.int32)
    assert (
        text_model._apply_blockwise_bidirectional_overlay(
            base_mask,
            short_token_types,
        )
        is base_mask
    )
