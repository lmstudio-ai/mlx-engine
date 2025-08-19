from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.utils import prepare_inputs

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import (
    load_and_parse_config,
    maybe_apply_quantization,
    load_and_filter_weights,
    prepare_components,
)
from mlx_engine.model_kit.vision_add_ons.load_utils import (
    load_processor,
    sanitize_weights,
)
from mlx_engine.utils.image_utils import convert_to_pil


class Qwen2VLVisionComponents(nn.Module):
    """Container for Qwen2-VL vision components."""

    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower


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

        # Load and parse configuration with correct classes
        config, config_dict = load_and_parse_config(
            model_path=model_path,
            model_config_class=ModelConfigClass,
            vision_config_class=VisionConfigClass,
            text_config_class=TextConfigClass,
        )

        # Create vision components container
        components = Qwen2VLVisionComponents(
            vision_tower=VisionTower(config.vision_config)
        )

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

        self.vision_tower = components.vision_tower
        self.config = config
        self.processor = load_processor(model_path)

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
        images = self._resize_images(images)

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

    def _resize_images(self, images, max_size=(1000, 1000)):
        """
        Resize large images without padding (preserves multi-image support).

        Args:
            images: List of PIL images
            max_size: Maximum dimensions (width, height)

        Returns:
            List of resized PIL images (no padding applied)
        """
        import PIL.Image
        import logging

        logger = logging.getLogger(__name__)
        resized_images = []

        for i, img in enumerate(images):
            original_size = (img.width, img.height)

            if img.width > max_size[0] or img.height > max_size[1]:
                # Resize while maintaining aspect ratio
                aspect_ratio = img.width / img.height
                if img.width > img.height:
                    new_width = max_size[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_size[1]
                    new_width = int(new_height * aspect_ratio)

                img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
                logger.info(
                    f"Image {i + 1}: Resized from {original_size} to {img.width}x{img.height}"
                )
            else:
                logger.info(f"Image {i + 1}: No resize needed {original_size}")

            resized_images.append(img)

        return resized_images
