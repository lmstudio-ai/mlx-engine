import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.utils import prepare_inputs

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_qwen_vision_addon
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize

logger = logging.getLogger(__name__)


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
        """Compute input_ids and embeddings for text with images."""

        # Convert prompt tokens to text
        detokenizer = self.processor.detokenizer
        detokenizer.reset()
        [detokenizer.add_token(token) for token in prompt_tokens]
        detokenizer.finalize()
        prompt = detokenizer.text

        # Convert images from base64
        images = convert_to_pil(images_b64)

        # Resize large images (without padding for multi-image support)
        images = custom_resize(images, should_pad=False)

        # Prepare inputs using mlx_vlm's prepare_inputs
        inputs = prepare_inputs(
            processor=self.processor,
            images=images,
            prompts=prompt,
            image_token_index=self.config.image_token_id,
            resize_shape=None,  # Let processor handle sizing
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]

        # Get prompt text embeddings
        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        # If no images, return input_ids and input_embeddings
        if pixel_values is None:
            return input_ids.squeeze(0), input_embeddings.squeeze(0)

        # Process through vision tower to get hidden states
        hidden_states = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        # Merge image features with text embeddings
        final_inputs_embeds = self.CombinedModel.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            input_embeddings,
            input_ids,
        )

        # Remove batch dimension
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
