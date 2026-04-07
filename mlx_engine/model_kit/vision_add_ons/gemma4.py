import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_vlm.models.gemma4 import (
    ModelConfig as Gemma4ModelConfig,
    TextConfig as Gemma4TextConfig,
    VisionConfig as Gemma4VisionConfig,
    VisionModel as Gemma4VisionTower,
)
from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder, masked_scatter
from mlx_vlm.utils import load_processor, sanitize_weights

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import (
    load_and_filter_weights,
    load_and_parse_config,
    maybe_apply_quantization,
    prepare_components,
)
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)

logger = logging.getLogger(__name__)


def _compute_prompt_per_layer_inputs(
    language_model: nn.Module,
    input_ids: mx.array,
    image_token_id: int,
    audio_token_id: int | None,
) -> mx.array | None:
    if not getattr(language_model, "hidden_size_per_layer_input", 0):
        return None

    image_mask_ids = input_ids == image_token_id
    audio_mask_ids = (
        input_ids == audio_token_id
        if audio_token_id is not None
        else mx.zeros_like(image_mask_ids)
    )
    text_mask = ~(image_mask_ids | audio_mask_ids)
    per_layer_inputs_tokens = mx.where(text_mask, input_ids, mx.zeros_like(input_ids))
    return language_model._get_per_layer_inputs(per_layer_inputs_tokens)


class Gemma4VisionComponents(nn.Module):
    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision


class Gemma4VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Gemma4 models.

    Gemma4's text model still applies `embed_scale` when `input_embeddings` are
    provided, so image features must be pre-divided by that scale before being
    scattered into the mixed prompt embeddings.
    """

    def __init__(self, model_path: Path):
        super().__init__()

        config, config_dict = load_and_parse_config(
            model_path=model_path,
            model_config_class=Gemma4ModelConfig,
            vision_config_class=Gemma4VisionConfig,
            text_config_class=Gemma4TextConfig,
        )

        components = Gemma4VisionComponents(
            vision_tower=Gemma4VisionTower(config.vision_config),
            embed_vision=MultimodalEmbedder(
                embedding_dim=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                eps=config.vision_config.rms_norm_eps,
            ),
        )

        processor = load_processor(model_path=model_path, add_detokenizer=True)
        vision_weights = load_and_filter_weights(model_path, components)
        vision_weights = sanitize_weights(
            components.vision_tower.__class__, vision_weights, config.vision_config
        )
        maybe_apply_quantization(components, config_dict, vision_weights)
        prepare_components(components, vision_weights)

        logger.info(f"Vision add-on loaded successfully from {model_path}")

        self.vision_tower = components.vision_tower
        self.embed_vision = components.embed_vision
        self.config = config
        self.processor = processor

    def clear_prediction_state(self, text_model: nn.Module) -> None:
        language_model = text_model.language_model.model
        reset = getattr(language_model, "reset_prompt_per_layer_input_state", None)
        if reset is not None:
            reset()

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """Compute input_ids and embeddings for text with images."""
        input_ids, pixel_values, _, _ = common_process_prompt_with_images(
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            processor=self.processor,
            config=self.config,
            max_size=max_size,
        )

        input_embeddings = text_model.language_model.model.embed_tokens(input_ids)

        image_features = self.vision_tower(pixel_values)
        image_features = self.embed_vision(image_features).astype(
            input_embeddings.dtype
        )

        # Gemma4TextModel applies embed_scale even when input_embeddings are provided.
        scaled_image_features = (
            image_features / text_model.language_model.model.embed_scale
        )

        image_mask = input_ids == self.config.image_token_id
        image_mask_expanded = mx.expand_dims(image_mask, -1)
        image_mask_expanded = mx.broadcast_to(
            image_mask_expanded, input_embeddings.shape
        )

        final_inputs_embeds = masked_scatter(
            input_embeddings, image_mask_expanded, scaled_image_features
        )

        prompt_per_layer_inputs = _compute_prompt_per_layer_inputs(
            text_model.language_model.model,
            input_ids,
            self.config.image_token_id,
            getattr(self.config, "audio_token_id", None),
        )
        if prompt_per_layer_inputs is not None:
            text_model.language_model.model.set_prompt_per_layer_inputs(
                prompt_per_layer_inputs
            )

        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
