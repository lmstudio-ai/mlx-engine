from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.models.pixtral import (
    VisionModel as PixtralVisionTower,
    ModelConfig as PixtralModelConfig,
    VisionConfig as PixtralVisionConfig,
    TextConfig as PixtralTextConfig,
    Model as PixtralCombinedModel,  # for merge_input_ids_with_image_features
)
from mlx_vlm.models.pixtral.pixtral import (
    LlavaMultiModalProjector as PixtralMultiModalProjector,
)

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon


class PixtralVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Pixtral model. Uses mlx-vlm vision components of Pixtral.
    """

    PIXTRAL_LOG_PREFIX = "PixtralVisionAddOn"

    def __init__(self, model_path: Path):
        """Initialize PixtralVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Load vision model components, configuration, and processor
        self.vision_tower, self.multi_modal_projector, self.config, self.processor = (
            load_vision_addon(
                model_path=model_path,
                model_config_class=PixtralModelConfig,
                vision_config_class=PixtralVisionConfig,
                text_config_class=PixtralTextConfig,
                vision_tower_class=PixtralVisionTower,
                multi_modal_projector_class=PixtralMultiModalProjector,
                log_prefix=self.PIXTRAL_LOG_PREFIX,
            )
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
    ) -> tuple[mx.array, mx.array]:
        """Compute embeddings for text with images."""
        input_ids, pixel_values, attention_mask, other_model_inputs = (
            common_process_prompt_with_images(
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                processor=self.processor,
                config=self.config,
            )
        )
        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Process image through vision tower
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.config.vision_feature_layer]

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = PixtralCombinedModel.merge_input_ids_with_image_features(
            self.config.image_token_index, image_features, input_embeddings, input_ids
        )
        # pixtral generation does not require input_ids, so we return an empty array
        return mx.array([]), final_inputs_embeds.squeeze(0)  # remove batch dimension
