"""Gemma 4 bidirectional-vision compatibility helpers for batched generation."""

from types import MethodType
from typing import Any

import mlx.core as mx


def is_unified_model_type(model_type: str | None) -> bool:
    return str(model_type or "").startswith("gemma4_unified")


def is_gemma4_model_type(model_type: str | None) -> bool:
    return str(model_type or "").startswith("gemma4")


def _language_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _model_type(model: Any) -> str | None:
    return getattr(_language_model(model), "model_type", None)


def is_unified_model(model: Any) -> bool:
    return is_unified_model_type(_model_type(model))


def uses_bidirectional_visual_attention(model: Any) -> bool:
    """Return true for Gemma 4 language models with visual bidirectional masks."""
    language_model = _language_model(model)
    if not is_gemma4_model_type(getattr(language_model, "model_type", None)):
        return False
    config = getattr(language_model, "config", None)
    return _get_config_value(config, "use_bidirectional_attention") == "vision"


def _get_config_value(config: dict | Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def config_uses_bidirectional_visual_attention(config: dict | Any) -> bool:
    """Config-level version used before the loaded model exists."""
    model_type = _get_config_value(config, "model_type")
    text_config = _get_config_value(config, "text_config") or config
    attention_mode = _get_config_value(text_config, "use_bidirectional_attention")
    return is_gemma4_model_type(model_type) and attention_mode == "vision"


def image_prefill_spans(
    model: Any,
    prompt_kwargs: dict,
    image_spans: list[Any],
    cached_prefix_len: int,
) -> list[tuple[int, int]] | None:
    """Return request-relative image runs that prefill must not split.

    Gemma 4 local-attention layers are bidirectional within each image block, so
    every contiguous image run must be visible in one model call. Token type ids
    are authoritative when present; prepared image spans are the fallback.
    """
    token_types = prompt_kwargs.get("mm_token_type_ids")
    if token_types is None:
        token_types = prompt_kwargs.get("token_type_ids")

    if isinstance(token_types, mx.array):
        values = token_types.reshape(-1).tolist()
        if is_gemma4_model_type(_model_type(model)) and 2 in values:
            raise ValueError("Gemma 4 video input is not supported by the MLX backend")
    else:
        values = None

    if not uses_bidirectional_visual_attention(model):
        return None

    if values is not None:
        spans = []
        image_start = None
        for index, value in enumerate(values):
            if value == 1 and image_start is None:
                image_start = index
            elif value != 1 and image_start is not None:
                spans.append((image_start, index))
                image_start = None
        if image_start is not None:
            spans.append((image_start, len(values)))
        return spans

    spans = []
    for span in image_spans:
        if span.start < cached_prefix_len < span.end:
            raise ValueError("A restored Gemma 4 prompt cache splits an image block")
        relative_start = span.start - cached_prefix_len
        relative_end = span.end - cached_prefix_len
        if relative_end > 0:
            spans.append((max(relative_start, 0), relative_end))
    return sorted(spans)


def prepare_cached_suffix_prompt_kwargs(prompt_kwargs: dict, key_len: int) -> dict:
    """Pad visual token-type ids so Gemma 4 masks can line up with cached keys."""
    prepared = prompt_kwargs
    for name in ("mm_token_type_ids", "token_type_ids"):
        token_type_ids = prepared.get(name)
        padded = _pad_visual_token_type_ids_to_key_len(token_type_ids, key_len)
        if padded is not token_type_ids:
            if prepared is prompt_kwargs:
                prepared = dict(prompt_kwargs)
            prepared[name] = padded
    return prepared


def patch_loaded_model(model: Any) -> None:
    """Match Transformers Gemma 4 visual masks for cached, chunked prefill."""
    language_model = _language_model(model)
    if not uses_bidirectional_visual_attention(language_model):
        return
    text_model = getattr(language_model, "model", language_model)
    if getattr(text_model, "_mlx_engine_gemma4_visual_mask_patch", False):
        return
    if not hasattr(text_model, "_apply_blockwise_bidirectional_overlay"):
        return

    # mlx-vlm requires square masks and applies the visual overlay to full
    # attention. Transformers keeps full attention causal and applies the
    # overlay only to sliding attention. This version also aligns cached suffix
    # query rows with the keys visible to the current layer.
    def _apply_blockwise_bidirectional_overlay(
        self,
        base_mask,
        mm_token_type_ids,
        window_size=None,
    ):
        if mm_token_type_ids is None:
            return base_mask
        key_len = base_mask.shape[-1]
        if mm_token_type_ids.shape[1] < key_len:
            return base_mask
        if mm_token_type_ids.shape[1] > key_len:
            mm_token_type_ids = mm_token_type_ids[:, -key_len:]

        block_sequence_ids = self._block_sequence_ids_for_mask(mm_token_type_ids)
        query_len = base_mask.shape[-2]
        query_block_sequence_ids = block_sequence_ids[:, -query_len:]
        query_blocks = mx.expand_dims(query_block_sequence_ids, -1)
        key_blocks = mx.expand_dims(block_sequence_ids, -2)
        same_block = (query_blocks != -1) & (query_blocks == key_blocks)
        if window_size is not None:
            query_positions = mx.arange(key_len - query_len, key_len)[:, None]
            key_positions = mx.arange(key_len)[None, :]
            same_block = same_block & (key_positions > query_positions - window_size)
        return base_mask | mx.expand_dims(same_block, 1)

    text_model._apply_blockwise_bidirectional_overlay = MethodType(
        _apply_blockwise_bidirectional_overlay,
        text_model,
    )

    if hasattr(text_model, "_make_masks"):
        from mlx_vlm.models.cache import create_causal_mask

        original_make_masks = text_model._make_masks

        def _make_masks(self, h, cache, mm_token_type_ids=None):
            if mm_token_type_ids is None:
                return original_make_masks(h, cache, mm_token_type_ids)
            if int(mx.sum(mm_token_type_ids == 2).item()) > 0:
                raise ValueError(
                    "Gemma 4 video input is not supported by the MLX backend"
                )

            has_image_tokens = int(mx.sum(mm_token_type_ids == 1).item()) > 0
            if not has_image_tokens or h.shape[1] <= 1:
                return original_make_masks(h, cache, mm_token_type_ids)

            masks = original_make_masks(h, cache, None)
            patched_sliding_mask = None
            patched_masks = []
            for layer, base_mask in zip(self.layers, masks):
                if layer.layer_type != "sliding_attention":
                    patched_masks.append(base_mask)
                    continue
                if patched_sliding_mask is None:
                    if isinstance(base_mask, str) and base_mask == "causal":
                        base_mask = create_causal_mask(
                            h.shape[1],
                            window_size=self.window_size,
                        )
                    patched_sliding_mask = self._apply_blockwise_bidirectional_overlay(
                        base_mask,
                        mm_token_type_ids,
                        window_size=self.window_size,
                    )
                patched_masks.append(patched_sliding_mask)
            return patched_masks

        text_model._make_masks = MethodType(_make_masks, text_model)

    text_model._mlx_engine_gemma4_visual_mask_patch = True


def _pad_visual_token_type_ids_to_key_len(
    token_type_ids: Any,
    key_len: int,
) -> Any:
    if not isinstance(token_type_ids, mx.array):
        return token_type_ids
    if key_len <= token_type_ids.shape[1]:
        return token_type_ids
    if int(mx.sum((token_type_ids == 1) | (token_type_ids == 2)).item()) == 0:
        return token_type_ids

    prefix = mx.zeros(
        (token_type_ids.shape[0], key_len - token_type_ids.shape[1]),
        dtype=token_type_ids.dtype,
    )
    return mx.concatenate([prefix, token_type_ids], axis=1)
