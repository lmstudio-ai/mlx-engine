import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.models.gemma3 import (
    VisionModel as Gemma3VisionTower,
    ModelConfig as Gemma3ModelConfig,
    VisionConfig as Gemma3VisionConfig,
    TextConfig as Gemma3TextConfig,
    Model as Gemma3CombinedModel,  # for prepare_inputs_for_multimodal
)
from mlx_vlm.models.gemma3.gemma3 import Gemma3MultiModalProjector

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon

logger = logging.getLogger(__name__)


class Gemma3VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Gemma3 model. Uses mlx-vlm vision components of Gemma3.
    """

    def __init__(self, model_path: Path):
        """Initialize Gemma3VisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Load vision model components, configuration, and processor
        self.vision_tower, self.multi_modal_projector, self.config, self.processor = (
            load_vision_addon(
                model_path=model_path,
                model_config_class=Gemma3ModelConfig,
                vision_config_class=Gemma3VisionConfig,
                text_config_class=Gemma3TextConfig,
                vision_tower_class=Gemma3VisionTower,
                multi_modal_projector_class=Gemma3MultiModalProjector,
                logger=logger,
            )
        )

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
        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        # Process image through vision tower
        hidden_state, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(input_embeddings.dtype),
            output_hidden_states=True,
        )

        # Format image features
        image_features = hidden_state.astype(pixel_values.dtype)
        image_features = self.multi_modal_projector(image_features)

        # Combine image and text embeddings
        final_inputs_embeds, _ = Gemma3CombinedModel.prepare_inputs_for_multimodal(
            self.config.hidden_size,
            self.config.pad_token_id,
            self.config.image_token_index,
            image_features,
            input_embeddings,
            input_ids,
            attention_mask,
        )
        # remove batch dimension
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
