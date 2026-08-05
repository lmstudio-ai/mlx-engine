from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.models.cache import create_causal_mask

from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.patches.gemma4 import (
    config_uses_bidirectional_visual_attention,
    image_prefill_sections,
    patch_loaded_model,
    prepare_cached_suffix_prompt_kwargs,
    uses_bidirectional_visual_attention,
)


class _Gemma4TextModel:
    window_size = 2
    layers = [
        SimpleNamespace(layer_type="full_attention"),
        SimpleNamespace(layer_type="sliding_attention"),
    ]

    def _block_sequence_ids_for_mask(self, mm_token_type_ids):
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        prev = mx.concatenate(
            [mx.zeros_like(is_vision[:, :1]), is_vision[:, :-1]],
            axis=1,
        )
        starts = is_vision & ~prev
        group_ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
        return mx.where(is_vision, group_ids, mx.zeros_like(group_ids) - 1)

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        del mm_token_type_ids
        return [
            create_causal_mask(
                h.shape[1],
                offset=getattr(cache[0], "offset", 0),
            ),
            create_causal_mask(
                h.shape[1],
                offset=getattr(cache[1], "offset", 0),
                window_size=self.window_size,
            ),
        ]


def _gemma4_model(text_model=None):
    if text_model is None:
        text_model = _Gemma4TextModel()
    language_model = SimpleNamespace(
        model_type="gemma4_text",
        config=SimpleNamespace(use_bidirectional_attention="vision"),
        model=text_model,
    )
    return SimpleNamespace(language_model=language_model)


def test_gemma4_cached_suffix_prompt_kwargs_pad_visual_token_types_to_key_len():
    """Restored Gemma4 visual suffix masks need token types for cached keys."""
    prompt_kwargs = {
        "mm_token_type_ids": mx.array([[0, 1, 0, 1]], dtype=mx.int32),
        "unchanged": "value",
    }

    prepared = prepare_cached_suffix_prompt_kwargs(prompt_kwargs, key_len=7)

    assert prepared is not prompt_kwargs
    assert prepared["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 0, 1]]
    assert prepared["unchanged"] == "value"


def test_gemma4_image_prompt_kwargs_without_cached_prefix_are_unchanged():
    prompt_kwargs = {
        "mm_token_type_ids": mx.array([[0, 1, 1, 0]], dtype=mx.int32),
    }

    prepared = prepare_cached_suffix_prompt_kwargs(prompt_kwargs, key_len=4)

    assert prepared is prompt_kwargs


def test_gemma4_suffix_visual_mask_patch_uses_query_rows_only():
    text_model = _Gemma4TextModel()
    patch_loaded_model(_gemma4_model(text_model))
    cache = SimpleNamespace(offset=5)
    token_types = mx.array([[0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=mx.int32)

    _, patched = text_model._make_masks(
        mx.zeros((1, 4, 4), dtype=mx.float32),
        [cache, cache],
        token_types,
    )
    causal_mask = create_causal_mask(4, offset=5, window_size=text_model.window_size)

    assert patched.shape == (1, 1, 4, 9)
    assert bool(patched[0, 0, 2, 8].item())
    assert not bool(causal_mask[2, 8].item())


def test_gemma4_mask_patch_matches_transformers_attention_topology():
    text_model = _Gemma4TextModel()
    patch_loaded_model(_gemma4_model(text_model))
    token_types = mx.array([[0, 1, 1, 1, 3]], dtype=mx.int32)

    full_mask, sliding_mask = text_model._make_masks(
        mx.zeros((1, 5, 4), dtype=mx.float32),
        [None, None],
        token_types,
    )

    # Full attention is causal even within an image block.
    assert not bool(full_mask[1, 2].item())
    assert bool(full_mask[3, 1].item())

    # Sliding attention is causal OR same-image-block, then windowed.
    assert bool(sliding_mask[0, 0, 1, 2].item())
    assert bool(sliding_mask[0, 0, 1, 3].item())
    assert not bool(sliding_mask[0, 0, 3, 1].item())
    assert not bool(sliding_mask[0, 0, 1, 4].item())


def test_gemma4_image_prefill_sections_are_cache_aligned_and_suffix_relative():
    sections = image_prefill_sections(
        _gemma4_model(),
        [
            PromptImageSpan(start=300, end=400, image_hash="cached"),
            PromptImageSpan(start=600, end=700, image_hash="first"),
            PromptImageSpan(start=1_100, end=1_200, image_hash="second"),
        ],
        cached_prefix_len=512,
    )

    assert sections == [(0, 256), (512, 768)]


def test_gemma4_image_prefill_sections_merge_shared_cache_chunks():
    sections = image_prefill_sections(
        _gemma4_model(),
        [
            PromptImageSpan(start=300, end=400, image_hash="first"),
            PromptImageSpan(start=500, end=600, image_hash="second"),
        ],
        cached_prefix_len=0,
    )

    assert sections == [(256, 768)]


def test_gemma4_bidirectional_visual_detection_accepts_top_and_text_config():
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model_type="gemma4_text",
            config=SimpleNamespace(use_bidirectional_attention="vision"),
        )
    )
    config = {
        "model_type": "gemma4",
        "text_config": {"use_bidirectional_attention": "vision"},
    }

    assert uses_bidirectional_visual_attention(model)
    assert config_uses_bidirectional_visual_attention(config)


def test_gemma4_bidirectional_visual_detection_rejects_non_bidir_config():
    model = SimpleNamespace(
        model_type="gemma4_text",
        config=SimpleNamespace(use_bidirectional_attention=None),
    )
    config = {
        "model_type": "gemma4",
        "text_config": {"use_bidirectional_attention": None},
    }

    assert not uses_bidirectional_visual_attention(model)
    assert not config_uses_bidirectional_visual_attention(config)
