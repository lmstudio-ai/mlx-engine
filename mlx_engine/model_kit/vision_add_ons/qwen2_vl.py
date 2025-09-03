import logging
from typing import Any, Tuple, Type
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import (
    load_and_parse_config,
    load_processor,
    load_and_filter_weights,
    sanitize_weights,
    maybe_apply_quantization,
    prepare_components,
)
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize

from mlx_vlm.utils import prepare_inputs

logger = logging.getLogger(__name__)


class QwenVisionComponents(nn.Module):
    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower


def load_qwen_vision_addon(
    model_path: Path,
    model_config_class: Any,
    vision_config_class: Any,
    text_config_class: Any,
    vision_tower_class: Type[nn.Module],
    logger: logging.Logger,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load Qwen2/2.5-VL vision add-on components, configuration, and processor.

    This is a specialized version of load_vision_addon for Qwen2/2.5-VL models,
    which only use a vision tower without a multi-modal projector.

    Args:
        model_path: Path to the model directory
        model_config_class: Configuration class for the model
        vision_config_class: Configuration class for vision component
        text_config_class: Configuration class for text component
        vision_tower_class: The vision tower model class
        logger: logging.Logger

    Returns:
        Tuple containing:
            - The vision tower module
            - The model configuration
            - The processor for handling images and text
    """
    # Load and parse configuration with correct classes
    config, config_dict = load_and_parse_config(
        model_path=model_path,
        model_config_class=model_config_class,
        vision_config_class=vision_config_class,
        text_config_class=text_config_class,
    )

    # Create vision components container (Qwen2/2.5-VL only use vision tower)
    components = QwenVisionComponents(
        vision_tower=vision_tower_class(config.vision_config)
    )

    # Load processor
    processor = load_processor(model_path=model_path, add_detokenizer=True)

    # Load and filter weights
    vision_weights = load_and_filter_weights(
        model_path=model_path,
        components=components,
    )

    # Sanitize vision weights
    vision_weights = sanitize_weights(
        components.vision_tower.__class__, vision_weights, config.vision_config
    )

    # Apply quantization if specified
    maybe_apply_quantization(
        components=components,
        config_dict=config_dict,
        vision_weights=vision_weights,
    )

    # Prepare components
    prepare_components(
        components=components,
        vision_weights=vision_weights,
    )

    logger.info(
        f"Qwen2/2.5-VL vision add-on loaded successfully from {model_path}",
    )

    return components.vision_tower, config, processor


class Qwen2_VLVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen2-VL and Qwen2.5-VL models.
    """

    def __init__(self, model_path: Path):
        """Initialize Qwen2_VLVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Determine model type from config to select appropriate classes
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            import json

            config_dict = json.load(f)
            model_type = config_dict.get("model_type")

        # Import appropriate classes based on model type
        if model_type == "qwen2_5_vl":
            from mlx_vlm.models.qwen2_5_vl import (
                VisionModel as VisionTower,
                ModelConfig as ModelConfigClass,
                VisionConfig as VisionConfigClass,
                TextConfig as TextConfigClass,
                Model as CombinedModel,
            )
        else:  # Default to qwen2_vl
            from mlx_vlm.models.qwen2_vl import (
                VisionModel as VisionTower,
                ModelConfig as ModelConfigClass,
                VisionConfig as VisionConfigClass,
                TextConfig as TextConfigClass,
                Model as CombinedModel,
            )

        # Store the combined model class for use in compute_embeddings
        self.CombinedModel = CombinedModel

        self.vision_tower, self.config, self.processor = load_qwen_vision_addon(
            model_path=model_path,
            model_config_class=ModelConfigClass,
            vision_config_class=VisionConfigClass,
            text_config_class=TextConfigClass,
            vision_tower_class=VisionTower,
            logger=logger,
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
    ) -> tuple[mx.array, mx.array]:
        """
        Compute input_ids and embeddings for text with images.
        """

        # Convert and resize images
        images = convert_to_pil(images_b64)
        images = custom_resize(images, should_pad=False)

        # Build prompt text
        tokens = (
            prompt_tokens if isinstance(prompt_tokens, list) else prompt_tokens.tolist()
        )
        prompt = self.processor.decode(tokens)

        # Prepare inputs
        inputs = prepare_inputs(
            processor=self.processor,
            images=images,
            prompts=prompt,
            image_token_index=self.config.image_token_id,
            resize_shape=None,
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs.get("image_grid_thw")

        # Get text embeddings - directly access the embed layer
        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        # If no images, return input_ids and input_embeddings
        if pixel_values is None:
            return input_ids.squeeze(0), input_embeddings.squeeze(0)

        # Ensure pixel_values are in the right format for vision tower
        if pixel_values.dtype != input_embeddings.dtype:
            pixel_values = pixel_values.astype(input_embeddings.dtype)

        # Process image through vision tower
        hidden_states = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Merge embeddings
        final_inputs_embeds = self.CombinedModel.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            input_embeddings,
            input_ids,
        )

        # Remove batch dimension
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
