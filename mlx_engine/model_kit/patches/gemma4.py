"""Gemma 4 bidirectional-vision compatibility helpers for batched generation."""

from types import MethodType
from typing import Any

import mlx.core as mx
from mlx_vlm.models.cache import create_causal_mask


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


def _get_token_type_ids(prompt_kwargs: dict) -> mx.array | None:
    token_types = prompt_kwargs.get("mm_token_type_ids")
    if token_types is None:
        token_types = prompt_kwargs.get("token_type_ids")
    return token_types


def image_prefill_runs(
    model: Any,
    prompt_kwargs: dict,
) -> list[tuple[int, int]] | None:
    """Return suffix-relative visual runs that prefill must not split.

    Gemma's token types are the source of truth for visual attention runs. Cache
    image spans can conservatively cover the whole prompt when image sentinels
    are unavailable, so they must remain separate from this prefill plan.
    """
    if not uses_bidirectional_visual_attention(model):
        return None

    token_types = _get_token_type_ids(prompt_kwargs)
    if token_types is None:
        return []

    # Materialize once while building the request's prefill plan. Each contiguous
    # type-1 run is one bidirectional image block and must stay in a single model
    # call; later scheduling uses only these Python-side ranges.
    image_runs: list[tuple[int, int]] = []
    image_start = None
    for index, token_type in enumerate(token_types.reshape(-1).tolist()):
        if token_type == 1:
            if image_start is None:
                image_start = index
        elif image_start is not None:
            image_runs.append((image_start, index))
            image_start = None
    if image_start is not None:
        image_runs.append((image_start, token_types.shape[1]))

    return image_runs


def prepare_cached_suffix_prompt_kwargs(prompt_kwargs: dict, key_len: int) -> dict:
    """Pad visual token types to line up with cached keys when present."""
    prepared = prompt_kwargs
    for name in ("mm_token_type_ids", "token_type_ids"):
        token_types = prompt_kwargs.get(name)
        if token_types is None:
            continue

        prefix_len = key_len - token_types.shape[1]
        if prefix_len == 0:
            continue

        if prepared is prompt_kwargs:
            prepared = dict(prompt_kwargs)
        prepared[name] = mx.concatenate(
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

    # Transformers 5.14.1 is the source of truth for Gemma 4 mask composition:
    # https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py#L2112-L2162
    # mlx-vlm requires square masks and applies the visual overlay to full
    # attention. Transformers keeps full attention causal and applies the
    # overlay only to sliding attention. This patch also aligns cached-suffix
    # query rows with the keys visible to each sliding-attention layer.
    original_make_masks = text_model._make_masks

    def _make_masks(self, h, cache, mm_token_type_ids=None):
        # The batcher omits token types from image-free slices, so non-None is a
        # known image slice and does not require a synchronous value scan here.
        if mm_token_type_ids is None:
            return original_make_masks(h, cache, None)

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
