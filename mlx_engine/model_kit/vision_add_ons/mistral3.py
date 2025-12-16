import logging
import math
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_vlm.models.mistral3 import (
    VisionModel as Mistral3VisionTower,
    ModelConfig as Mistral3ModelConfig,
    VisionConfig as Mistral3VisionConfig,
    TextConfig as Mistral3TextConfig,
    Model as Mistral3CombinedModel,
)
from mlx_vlm.models.mistral3.mistral3 import Mistral3MultiModalProjector
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon

logger = logging.getLogger(__name__)


class Mistral3VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Mistral3 models.
    """

    def __init__(self, model_path: Path):
        """Initialize Mistral3VisionAddOn with vision components loaded from the given path."""
        super().__init__()
        self.model_path = Path(model_path)

        processor_kwargs = {}
        if self._is_lmstudio_mistral_3_2_small():
            processor_kwargs = {
                "patch_size": 14,
                "spatial_merge_size": 2,
            }
            logger.info(
                "Detected LM Studio Mistral Small 3.2 model. "
                f"Using custom processor kwargs: {processor_kwargs}"
            )

        self.vision_tower, self.multi_modal_projector, self.config, self.processor = (
            load_vision_addon(
                model_path=model_path,
                model_config_class=Mistral3ModelConfig,
                vision_config_class=Mistral3VisionConfig,
                text_config_class=Mistral3TextConfig,
                vision_tower_class=Mistral3VisionTower,
                multi_modal_projector_class=Mistral3MultiModalProjector,
                logger=logger,
                processor_kwargs=processor_kwargs,
            )
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Compute embeddings for text with images.

        This method is heavily based on the mlx-vlm's mistral3 `get_input_embeddings`
        https://github.com/Blaizzy/mlx-vlm/blob/2c3014fd40962bd5320ad611502e7e26cae08926/mlx_vlm/models/mistral3/mistral3.py#L240-L279
        """

        input_ids, pixel_values, attention_mask, other_model_inputs = (
            common_process_prompt_with_images(
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                processor=self.processor,
                config=self.config,
                max_size=max_size,
            )
        )

        image_sizes_list = other_model_inputs["image_sizes"]
        image_sizes = mx.array(image_sizes_list)

        if pixel_values is None:
            return text_model.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = text_model.language_model.model.embed_tokens(input_ids)

        # Get the output hidden states from the vision model
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Pass pixel_values as list of images, as each image is individually run through conv2d and position encoding
        # Reference code from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py#L479C9-L479C21
        # and mistral_inference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py#L85
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.config.vision_feature_layer]

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = Mistral3CombinedModel.merge_input_ids_with_image_features(
            self.config.image_token_index, image_features, inputs_embeds, input_ids
        )
        if input_ids.shape[1] == final_inputs_embeds.shape[1]:
            return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
        # Return fake input_ids b/c the original lmstudio-community MLX upload had an incorrect
        # processor_config.json that caused input_ids have extra placeholder image tokens.
        return mx.array(
            [0] * final_inputs_embeds.squeeze(0).shape[0]
        ), final_inputs_embeds.squeeze(0)
    
    def _is_lmstudio_mistral_3_2_small(self) -> bool:
        return "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX" in str(
            self.model_path
        )
