import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx_vlm

from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.batched_vision.qwen_mrope import (
    apply_qwen_image_mrope_state,
)
from mlx_engine.utils.image_utils import convert_to_pil


@dataclass
class PreparedPrompt:
    """Prompt tokens plus optional mlx-vlm multimodal processor inputs."""

    prompt_input_ids: list[int]
    raw_inputs: dict[str, Any] | None
    image_spans: list[PromptImageSpan]


def prepare_prompt_inputs(
    *,
    prompt_tokens: list[int],
    images_b64: list[str] | None,
    tokenizer,
    processor,
    config: dict,
) -> PreparedPrompt:
    """Prepare one request for local batched VLM prompt processing."""
    if len(prompt_tokens) == 0:
        prompt_tokens = _tokenize(tokenizer, " ")

    if not images_b64:
        return PreparedPrompt(
            prompt_input_ids=list(prompt_tokens),
            raw_inputs=None,
            image_spans=[],
        )

    # Request prep runs on the cache I/O thread before generation insertion.
    prompt = tokenizer.decode(prompt_tokens) or " "
    images = convert_to_pil(images_b64)
    image_token_index = get_image_token_index(config)
    raw_inputs = mlx_vlm.prepare_inputs(
        processor=processor,
        images=images,
        prompts=prompt,
        image_token_index=image_token_index,
        resize_shape=None,
    )
    prompt_input_ids = raw_inputs["input_ids"].squeeze(0).tolist()
    image_hashes = [_hash_prompt_image(image) for image in images]
    return PreparedPrompt(
        prompt_input_ids=prompt_input_ids,
        raw_inputs=raw_inputs,
        image_spans=_get_image_spans(
            prompt_input_ids,
            image_hashes,
            image_token_index,
        ),
    )


def build_prompt_kwargs(model, prepared_prompt: PreparedPrompt) -> dict:
    """Build model kwargs for a full prompt prefill."""
    if prepared_prompt.raw_inputs is None:
        input_ids = mx.array(prepared_prompt.prompt_input_ids, dtype=mx.int32)[None, :]
        embedding_output = model.get_input_embeddings(input_ids)
        return {
            key: value
            for key, value in embedding_output.to_dict().items()
            if value is not None
        }

    raw_inputs = prepared_prompt.raw_inputs
    input_ids = raw_inputs["input_ids"]
    pixel_values = raw_inputs.get("pixel_values")
    attention_mask = raw_inputs.get("attention_mask")
    data_kwargs = {
        key: value
        for key, value in raw_inputs.items()
        if key not in {"input_ids", "pixel_values", "attention_mask"}
    }
    embedding_output = model.get_input_embeddings(
        input_ids,
        pixel_values,
        mask=attention_mask,
        **data_kwargs,
    )
    apply_qwen_image_mrope_state(
        model,
        input_ids=input_ids,
        image_grid_thw=raw_inputs.get("image_grid_thw"),
    )
    prompt_kwargs = {
        **data_kwargs,
        **{
            key: value
            for key, value in embedding_output.to_dict().items()
            if value is not None
        },
    }
    _add_language_model_rope_state(model, prompt_kwargs)
    return prompt_kwargs


def build_cached_prompt_kwargs(
    model,
    prepared_prompt: PreparedPrompt,
    cached_prefix_len: int,
    rope_deltas: Any | None,
) -> dict:
    """Build model kwargs for the uncached suffix after a prefix restore."""
    prompt_input_ids = prepared_prompt.prompt_input_ids[cached_prefix_len:]
    if prepared_prompt.raw_inputs is not None:
        prompt_kwargs = build_prompt_kwargs(model, prepared_prompt)
        # Keep full-prompt model side state, but prefill only the suffix.
        prompt_kwargs["inputs_embeds"] = prompt_kwargs["inputs_embeds"][
            :, cached_prefix_len:
        ]
        if "position_ids" in prompt_kwargs:
            # MRoPE image positions are request-local; slice them with embeds.
            prompt_kwargs["position_ids"] = prompt_kwargs["position_ids"][
                :, :, cached_prefix_len:
            ]
        return prompt_kwargs

    input_ids = mx.array(prompt_input_ids, dtype=mx.int32)[None, :]
    embedding_output = model.get_input_embeddings(input_ids)
    prompt_kwargs = {
        key: value
        for key, value in embedding_output.to_dict().items()
        if value is not None
    }

    # Prefix restores carry the tiny RoPE delta side state in memory.
    if rope_deltas is not None and prompt_kwargs.get("rope_deltas") is None:
        prompt_kwargs["rope_deltas"] = rope_deltas

    return prompt_kwargs


def _add_language_model_rope_state(model, prompt_kwargs: dict) -> None:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return

    position_ids = getattr(language_model, "_position_ids", None)
    if position_ids is not None:
        prompt_kwargs["position_ids"] = position_ids

    rope_deltas = getattr(language_model, "_rope_deltas", None)
    if rope_deltas is not None:
        prompt_kwargs["rope_deltas"] = rope_deltas


def get_image_token_index(config: dict) -> int | None:
    for value in (
        config.get("image_token_index"),
        config.get("image_token_id"),
        config.get("media_placeholder_token_id"),
        config.get("vision_config", {}).get("image_token_id"),
    ):
        if value is not None:
            return value
    return None


def _tokenize(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    if isinstance(ids, int):
        return [ids]
    return ids


def _hash_prompt_image(image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode())
    digest.update(f"{image.size[0]}x{image.size[1]}".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _get_image_spans(
    prompt_input_ids: list[int],
    image_hashes: list[str],
    image_token_index: int | None,
) -> list[PromptImageSpan]:
    if not image_hashes:
        return []

    if image_token_index is None:
        # Some processors do not expose a stable image sentinel; keep the
        # cache correct by making the whole prompt image-dependent.
        return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

    token_spans = []
    span_start = None
    for i, token_id in enumerate(prompt_input_ids):
        if token_id == image_token_index:
            if span_start is None:
                span_start = i
        elif span_start is not None:
            token_spans.append((span_start, i))
            span_start = None
    if span_start is not None:
        token_spans.append((span_start, len(prompt_input_ids)))

    if len(token_spans) != len(image_hashes):
        # Mismatched processor output is rare, but wrong image reuse is worse
        # than missing a cache hit.
        return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

    return [
        PromptImageSpan(start, end, image_hash)
        for (start, end), image_hash in zip(token_spans, image_hashes)
    ]
