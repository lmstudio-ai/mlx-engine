import mlx.core as mx
from mlx import nn

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize

from mlx_vlm.utils import prepare_inputs


def compute_qwen_vl_embeddings(
    addon: BaseVisionAddOn,
    text_model: nn.Module,
    prompt_tokens: mx.array,
    images_b64: list[str],
    qwen_vl_version: int,
    max_size: tuple[int, int] | None,
) -> tuple[mx.array, mx.array]:
    """
    Compute input_ids and embeddings for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models.

    Args:
        addon: Vision add-on instance containing vision tower, config, and processor
        text_model: Text model for embedding tokens
        prompt_tokens: Input prompt tokens
        images_b64: List of base64-encoded images
        qwen_vl_version: Version number (2 for Qwen2/2.5-VL, 3 for Qwen3-VL)
        max_size: Maximum image size as (width, height) tuple. If None, no resizing.

    Returns:
        Tuple of (input_ids, final_embeddings) with batch dimension removed
    """

    # Convert and resize images
    images = convert_to_pil(images_b64)
    images = custom_resize(images, max_size=max_size, should_pad=False)

    # Build prompt text
    tokens = (
        prompt_tokens if isinstance(prompt_tokens, list) else prompt_tokens.tolist()
    )
    prompt = addon.processor.decode(tokens)

    # Prepare inputs
    inputs = prepare_inputs(
        processor=addon.processor,
        images=images,
        prompts=prompt,
        image_token_index=addon.config.image_token_id,
        resize_shape=None,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs.get("image_grid_thw")

    # Get text embeddings
    input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

    # If no images, return input_ids and input_embeddings
    if pixel_values is None:
        return input_ids.squeeze(0), input_embeddings.squeeze(0)

    # Ensure pixel values are in the right format for vision tower
    if pixel_values.dtype != input_embeddings.dtype:
        pixel_values = pixel_values.astype(input_embeddings.dtype)

    # Process image through vision tower and merge embeddings
    if qwen_vl_version == 2:
        hidden_states = addon.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        final_inputs_embeds = addon.model_cls.merge_input_ids_with_image_features(
            addon.config.image_token_id,
            addon.config.video_token_id,
            hidden_states,
            input_embeddings,
            input_ids,
        )
    elif qwen_vl_version == 3:
        hidden_states, _ = addon.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        final_inputs_embeds, _ = addon.model_cls.merge_input_ids_with_image_features(
            hidden_states,
            input_embeddings,
            input_ids,
            addon.config.image_token_id,
            addon.config.video_token_id,
        )
    else:
        raise ValueError(f"Invalid Qwen-VL version: {qwen_vl_version}")

    # Remove batch dimension
    return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
