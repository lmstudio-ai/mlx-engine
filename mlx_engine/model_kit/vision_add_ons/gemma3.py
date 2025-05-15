import glob
import json
from typing import List

from mlx import nn

from mlx_vlm.models.gemma3 import (
    VisionModel as Gemma3VisionTower,
    ModelConfig as Gemma3ModelConfig,
    VisionConfig as Gemma3VisionConfig,
    TextConfig as Gemma3TextConfig,
    Model as Gemma3CombinedModel,  # for prepare_inputs_for_multimodal
)
from mlx_vlm.models.gemma3.gemma3 import Gemma3MultiModalProjector
from mlx_vlm.utils import sanitize_weights, load_processor, get_class_predicate

from pathlib import Path
import mlx.core as mx

from mlx_engine.logging import log_info
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn


class Gemma3VisionAddOn(BaseVisionAddOn, nn.Module):
    """
    Vision add-on for Gemma3 model. Uses mlx-vlm vision components of Gemma3
    """

    GEMMA3_LOG_PREFIX = "Gemma3VisionAddOn"

    def __init__(self, model_path: Path):
        super().__init__()
        config_dict = json.loads((model_path / "config.json").read_text())
        self.config = Gemma3ModelConfig.from_dict(config_dict)
        self.config.vision_config = Gemma3VisionConfig.from_dict(
            self.config.vision_config
        )
        self.config.text_config = Gemma3TextConfig.from_dict(self.config.text_config)
        self.vision_tower = Gemma3VisionTower(self.config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(self.config)
        self.processor = load_processor(model_path=model_path, add_detokenizer=True)
        # load the weights for the vision tower
        # ref: https://github.com/Blaizzy/mlx-vlm/blob/d2391123cabac313729f9a2a8d57d396e2592f20/mlx_vlm/utils.py#L147
        # and https://github.com/Blaizzy/mlx-vlm/blob/d2391123cabac313729f9a2a8d57d396e2592f20/mlx_vlm/models/gemma3/gemma3.py#L86-L87
        weight_files = glob.glob(str(model_path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"Failed to load Gemma3 vision model: {model_path} does not contain any safetensors files"
            )
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        # filter out everything but weights with keys that start with "vision_tower" or "multi_modal_projector"
        weights = {
            k: v
            for k, v in weights.items()
            if k.startswith("vision_tower") or k.startswith("multi_modal_projector")
        }
        weights = sanitize_weights(
            Gemma3VisionTower, weights, self.config.vision_config
        )
        # perform jit quantization if needed
        if (quantization := config_dict.get("quantization", None)) is not None:
            class_predicate = get_class_predicate(skip_vision=False, weights=weights)
            nn.quantize(
                self,
                **quantization,
                class_predicate=class_predicate,
            )

        # load weights using nn.Module method
        self.load_weights(list(weights.items()))
        # hardcode lazy loading to false for now, always load weights to memory here
        lazy = False
        if not lazy:
            mx.eval(self.parameters())

        self.eval()
        log_info(
            prefix=self.GEMMA3_LOG_PREFIX,
            message=f"Gemma3 vision model loaded successfully from {model_path}",
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: List[str],
    ) -> mx.array:
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
        return final_inputs_embeds.squeeze(0)  # remove batch dimension
