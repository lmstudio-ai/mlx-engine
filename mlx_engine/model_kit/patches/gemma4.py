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
    """Return true for Gemma4-family language models with visual bidir masks."""
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


def visual_prefill_prefix_len(
    model: Any,
    prompt_kwargs: dict,
    image_spans: list[Any],
    cached_prefix_len: int,
) -> int | None:
    """Return the visual prefix Gemma4 bidir masks need in one model call.

    Gemma4 derives its bidirectional visual attention overlay from token-type ids
    visible to the current language-model call. Keep the first prefill anchored
    at the current suffix start through the last visual token; trailing text can
    go back to ordinary chunked prefill.
    """
    if not uses_bidirectional_visual_attention(model):
        return None

    token_types = prompt_kwargs.get("mm_token_type_ids")
    if token_types is None:
        token_types = prompt_kwargs.get("token_type_ids")
    if not isinstance(token_types, mx.array):
        # Token types are authoritative when present. Image spans are a fallback
        # for processor/model combinations that omit them but still carry images.
        last_visual_end = max((span.end for span in image_spans), default=0)
        relative_visual_end = last_visual_end - cached_prefix_len
        return relative_visual_end if relative_visual_end > 0 else None

    values = token_types.reshape(-1).tolist()
    last_visual_idx = -1
    for idx, value in enumerate(values):
        if value in (1, 2):
            last_visual_idx = idx

    return None if last_visual_idx < 0 else last_visual_idx + 1


def prepare_cached_suffix_prompt_kwargs(prompt_kwargs: dict, key_len: int) -> dict:
    """Pad visual token-type ids so Gemma4 masks can line up with cached keys."""
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
    """Patch loaded Gemma4 bidir models for cached visual suffix masks."""
    language_model = _language_model(model)
    if not uses_bidirectional_visual_attention(language_model):
        return
    text_model = getattr(language_model, "model", language_model)
    if getattr(text_model, "_mlx_engine_suffix_visual_mask_patch", False):
        return
    if not hasattr(text_model, "_apply_blockwise_bidirectional_overlay"):
        return

    # Upstream builds Gemma4's visual overlay from token types over the key
    # length. With a restored prefix, only the suffix is queried; with sliding
    # layers, only a recent key window is visible. Compare current query rows
    # against the key rows covered by this layer's mask.
    def _apply_blockwise_bidirectional_overlay(self, base_mask, mm_token_type_ids):
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
        q_blocks = mx.expand_dims(query_block_sequence_ids, -1)
        k_blocks = mx.expand_dims(block_sequence_ids, -2)
        same_block = (q_blocks != -1) & (q_blocks == k_blocks)
        return base_mask | mx.expand_dims(same_block, 1)

    text_model._apply_blockwise_bidirectional_overlay = MethodType(
        _apply_blockwise_bidirectional_overlay,
        text_model,
    )
    text_model._mlx_engine_suffix_visual_mask_patch = True


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
