import logging
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_vlm.models.lfm2_vl import (
    Model as LFM2VlModel,
)
from mlx_vlm.models.lfm2_vl import (
    ModelConfig as LFM2VlModelConfig,
)
from mlx_vlm.models.lfm2_vl import (
    TextConfig as LFM2VlTextConfig,
)
from mlx_vlm.models.lfm2_vl import (
    VisionConfig as LFM2VlVisionConfig,
)
from mlx_vlm.models.lfm2_vl import (
    VisionModel as LFM2VlVisionTower,
)
from mlx_vlm.models.lfm2_vl.lfm2_vl import (
    Lfm2VlMultiModalProjector,
    PixelUnshuffleBlock,
)

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)

logger = logging.getLogger(__name__)


class LFM2VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for LFM2 models.
    """

    def __init__(self, model_path: Path):
        """Initialize LFM2VisionAddOn with vision components loaded from the given path."""
        super().__init__()

        self.vision_tower, self.multi_modal_projector, self.config, self.processor = (
            load_vision_addon(
                model_path=model_path,
                model_config_class=LFM2VlModelConfig,
                vision_config_class=LFM2VlVisionConfig,
                text_config_class=LFM2VlTextConfig,
                vision_tower_class=LFM2VlVisionTower,
                multi_modal_projector_class=Lfm2VlMultiModalProjector,
                logger=logger,
            )
        )

        # this particular block comes from
        # https://github.com/Blaizzy/mlx-vlm/blob/f02d63e8f5b521e8c75f129a63d2660efd132693/mlx_vlm/models/lfm2_vl/lfm2_vl.py#L102-L105
        if self.config.downsample_factor > 1:
            self.pixel_unshuffle = PixelUnshuffleBlock(self.config.downsample_factor)
        else:
            self.pixel_unshuffle = nn.Identity()

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Compute embeddings for text with images.

        This method is heavily based on the mlx-vlm's lfm2_vl `get_input_embeddings`
        https://github.com/Blaizzy/mlx-vlm/blob/f02d63e8f5b521e8c75f129a63d2660efd132693/mlx_vlm/models/lfm2_vl/lfm2_vl.py#L110-L150
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

        # Get the input embeddings from the language model
        inputs_embeds = text_model.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return inputs_embeds

        spatial_shapes = other_model_inputs["spatial_shapes"]
        pixel_attention_mask = other_model_inputs["pixel_attention_mask"]

        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values, output_hidden_states=True, spatial_shapes=spatial_shapes
        )

        img_feature_lengths = pixel_attention_mask.sum(axis=1).tolist()
        image_features = []

        for img_idx in range(hidden_states.shape[0]):
            feature = hidden_states[img_idx]

            feature = feature[: img_feature_lengths[img_idx], :][None, ...]

            feature_org_h, feature_org_w = spatial_shapes[img_idx]
            feature = feature.reshape(1, feature_org_h, feature_org_w, -1)
            feature = self.pixel_unshuffle(feature)

            img_embedding = self.multi_modal_projector(feature)

            img_embedding = img_embedding.reshape(-1, img_embedding.shape[-1])
            image_features.append(img_embedding)

        image_features = mx.concatenate(image_features, axis=0)

        final_inputs_embeds = LFM2VlModel.merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, self.config.image_token_index
        )

        if input_ids.shape[1] == final_inputs_embeds.shape[1]:
            return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
        return input_ids, final_inputs_embeds
