import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize

from mlx_vlm.models.qwen3_vl import (
    VisionModel as Qwen3VLVisionTower,
    ModelConfig as Qwen3VLModelConfig,
    VisionConfig as Qwen3VLVisionConfig,
    TextConfig as Qwen3VLTextConfig,
    Model as Qwen3VLModel,
)
from mlx_vlm.utils import prepare_inputs

logger = logging.getLogger(__name__)


class Qwen3_VLVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen3-VL Dense models.
    """

    def __init__(self, model_path: Path):
        """Initialize Qwen3_VLVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Store the model class for use in compute_embeddings
        self.model_cls = Qwen3VLModel

        # Load vision components
        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=Qwen3VLModelConfig,
            vision_config_class=Qwen3VLVisionConfig,
            text_config_class=Qwen3VLTextConfig,
            vision_tower_class=Qwen3VLVisionTower,
            multi_modal_projector_class=None,
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

        # Get text embeddings
        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        # If no images, return input_ids and input_embeddings
        if pixel_values is None:
            return input_ids.squeeze(0), input_embeddings.squeeze(0)

        # Ensure pixel values are in the right format for vision tower
        if pixel_values.dtype != input_embeddings.dtype:
            pixel_values = pixel_values.astype(input_embeddings.dtype)

        # Process image through vision tower
        hidden_states, _ = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Merge embeddings
        final_inputs_embeds, _ = self.model_cls.merge_input_ids_with_image_features(
            hidden_states,
            input_embeddings,
            input_ids,
            self.config.image_token_id,
            self.config.video_token_id,
        )

        # Remove batch dimension
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
