from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.models.gemma3n import (
    VisionModel as Gemma3nVisionTower,
    ModelConfig as Gemma3nModelConfig,
    VisionConfig as Gemma3nVisionConfig,
    TextConfig as Gemma3nTextConfig,
    Model as Gemma3nCombinedModel,
)
from mlx_vlm.models.gemma3n.gemma3n import Gemma3nMultimodalEmbedder
from mlx_vlm.utils import sanitize_weights, load_processor
from mlx_engine.logging import log_info

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.model_kit.vision_add_ons.load_utils import (
    load_and_filter_weights,
    load_and_parse_config,
    maybe_apply_quantization,
    prepare_components,
)


class Gemma3nVisionComponents(nn.Module):
    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision


class Gemma3nVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Gemma3n model. Uses mlx-vlm vision components of Gemma3n.
    """

    GEMMA3N_LOG_PREFIX = "Gemma3nVisionAddOn"

    def __init__(self, model_path: Path):
        """Initialize Gemma3nVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        config, config_dict = load_and_parse_config(
            model_path=model_path,
            model_config_class=Gemma3nModelConfig,
            vision_config_class=Gemma3nVisionConfig,
            text_config_class=Gemma3nTextConfig,
        )

        components = Gemma3nVisionComponents(
            vision_tower=Gemma3nVisionTower(config.vision_config),
            embed_vision=Gemma3nMultimodalEmbedder(
                config.vision_config, config.text_config
            ),
        )
        processor = load_processor(model_path=model_path, add_detokenizer=True)
        vision_weights = load_and_filter_weights(model_path, components)
        vision_weights = sanitize_weights(
            components.vision_tower.__class__, vision_weights, config.vision_config
        )
        maybe_apply_quantization(components, config_dict, vision_weights)
        prepare_components(components, vision_weights)

        log_info(
            prefix=self.GEMMA3N_LOG_PREFIX,
            message=f"Vision add-on loaded successfully from {model_path}",
        )

        self.vision_tower = components.vision_tower
        self.embed_vision = components.embed_vision
        self.config = config
        self.processor = processor

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
    ) -> tuple[mx.array, mx.array]:
        """Compute input_ids and embeddings for text with images."""
        input_ids, pixel_values, attention_mask, other_model_inputs = (
            common_process_prompt_with_images(
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                processor=self.processor,
                config=self.config,
            )
        )
        assert input_ids is not None

        # See mlx_vlm.models.gemma3n.gemma3n.Model.get_input_embeddings
        # This implementation was based on commit mlx-vlm commit ebafa5a789ed1a8e050b8366ae4e845dbe640b90
        # It differs slightly from mlx-vlm in the bounds on the vision_mask.
        # However, the two calculations should be equivalent (vision vocab offset + size) == audio vocab offset
        inputs_embeds = text_model.model.language_model.embed_tokens(input_ids)
        vision_mask = mx.logical_and(
            input_ids >= self.embed_vision.vocab_offset,
            input_ids < self.embed_vision.vocab_offset + self.embed_vision.vocab_size,
        )
        dummy_vision_token_id = (
            self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
        )
        vision_tokens = mx.where(vision_mask, input_ids, dummy_vision_token_id)
        vision_embeds_flat = self.embed_vision(input_ids=vision_tokens)
        inputs_embeds = mx.where(
            vision_mask[..., None], vision_embeds_flat, inputs_embeds
        )

        # Process image through vision tower, then embed into language model space
        image_features = Gemma3nCombinedModel.get_image_features(
            pixel_values,
            self.vision_tower,
            self.config,
            self.embed_vision,
        )

        # Construct mask that matches image embedding locations
        special_modality_mask = mx.expand_dims(
            input_ids == self.config.image_token_id, -1
        )
        special_modality_mask = mx.broadcast_to(
            special_modality_mask, inputs_embeds.shape
        )

        # Construct embeddings with image and text tokens interleaved per special modality mask
        final_inputs_embeds = Gemma3nCombinedModel.merge_multimodal_and_text(
            inputs_embeds, image_features, special_modality_mask, "image"
        )
        # remove batch dimension
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
