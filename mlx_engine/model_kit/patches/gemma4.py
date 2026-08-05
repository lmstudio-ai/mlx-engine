"""Gemma 4 bidirectional-vision compatibility helpers for batched generation."""

from types import MethodType
from typing import Any

import mlx.core as mx

from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan


def is_unified_model_type(model_type: str | None) -> bool:
    return model_type is not None and model_type.startswith("gemma4_unified")


def is_gemma4_model_type(model_type: str | None) -> bool:
    return model_type is not None and model_type.startswith("gemma4")


def _language_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def uses_bidirectional_visual_attention(model: Any) -> bool:
    """Return true for Gemma 4 language models with visual bidirectional masks."""
    language_model = _language_model(model)
    return (
        is_gemma4_model_type(getattr(language_model, "model_type", None))
        and getattr(language_model.config, "use_bidirectional_attention", None)
        == "vision"
    )


def config_uses_bidirectional_visual_attention(config: dict) -> bool:
    """Config-level version used before the loaded model exists."""
    return (
        is_gemma4_model_type(config.get("model_type"))
        and config["text_config"].get("use_bidirectional_attention") == "vision"
    )


def _contains_token_type(token_types: mx.array | None, value: int) -> bool:
    return token_types is not None and bool(mx.any(token_types == value).item())


def image_prefill_spans(
    model: Any,
    prompt_kwargs: dict,
    image_spans: list[PromptImageSpan],
    cached_prefix_len: int,
) -> list[tuple[int, int]] | None:
    """Return suffix-relative image spans that prefill must not split.

    Prepared image spans and Gemma 4 token types come from the same expanded
    image-token runs. Cache restore rejects positions inside those spans before
    constructing the suffix, so every remaining span starts at or after it.
    """
    if not uses_bidirectional_visual_attention(model):
        return None

    token_types = prompt_kwargs.get("mm_token_type_ids")
    if _contains_token_type(token_types, 2):
        raise ValueError("Gemma 4 video input is not supported by the MLX backend")

    return [
        (span.start - cached_prefix_len, span.end - cached_prefix_len)
        for span in image_spans
        if span.start >= cached_prefix_len
    ]


def prepare_cached_suffix_prompt_kwargs(prompt_kwargs: dict, key_len: int) -> dict:
    """Pad image token types so a Gemma 4 suffix lines up with cached keys."""
    token_types = prompt_kwargs.get("mm_token_type_ids")
    if token_types is None:
        return prompt_kwargs

    prefix_len = key_len - token_types.shape[1]
    if prefix_len == 0 or not _contains_token_type(token_types, 1):
        return prompt_kwargs

    prepared = dict(prompt_kwargs)
    prepared["mm_token_type_ids"] = mx.concatenate(
        [
            mx.zeros(
                (token_types.shape[0], prefix_len),
                dtype=token_types.dtype,
            ),
            token_types,
        ],
        axis=1,
    )
    return prepared


def patch_loaded_model(model: Any) -> None:
    """Match Transformers Gemma 4 visual masks for cached, chunked prefill."""
    language_model = _language_model(model)
    if not uses_bidirectional_visual_attention(language_model):
        return

    text_model = language_model.model
    if getattr(text_model, "_mlx_engine_gemma4_visual_mask_patch", False):
        return

    from mlx_vlm.models.cache import create_causal_mask

    original_make_masks = text_model._make_masks

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        if not _contains_token_type(mm_token_type_ids, 1):
            return original_make_masks(h, cache, mm_token_type_ids)

        # Transformers keeps full attention causal and adds the bidirectional
        # image-block overlay only to sliding attention.
        masks = original_make_masks(h, cache, None)
        sliding_mask = next(
            mask
            for layer, mask in zip(self.layers, masks)
            if layer.layer_type == "sliding_attention"
        )
        if isinstance(sliding_mask, str):
            sliding_mask = create_causal_mask(
                h.shape[1],
                window_size=self.window_size,
            )

        key_len = sliding_mask.shape[-1]
        query_len = sliding_mask.shape[-2]
        visible_token_types = mm_token_type_ids[:, -key_len:]
        block_ids = self._block_sequence_ids_for_mask(visible_token_types)
        query_blocks = mx.expand_dims(block_ids[:, -query_len:], -1)
        key_blocks = mx.expand_dims(block_ids, -2)
        same_block = (query_blocks != -1) & (query_blocks == key_blocks)

        query_positions = mx.arange(key_len - query_len, key_len)[:, None]
        key_positions = mx.arange(key_len)[None, :]
        within_window = key_positions > query_positions - self.window_size
        sliding_mask = sliding_mask | mx.expand_dims(same_block & within_window, 1)

        return [
            sliding_mask if layer.layer_type == "sliding_attention" else mask
            for layer, mask in zip(self.layers, masks)
        ]

    text_model._make_masks = MethodType(_make_masks, text_model)
    text_model._mlx_engine_gemma4_visual_mask_patch = True
