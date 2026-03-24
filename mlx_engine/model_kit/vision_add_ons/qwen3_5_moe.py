import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import compute_qwen_vl_embeddings

from mlx_vlm.models.qwen3_5_moe import (
    VisionModel as Qwen3_5MoEVisionTower,
    ModelConfig as Qwen3_5MoEModelConfig,
    VisionConfig as Qwen3_5MoEVisionConfig,
    TextConfig as Qwen3_5MoETextConfig,
    Model as Qwen3_5MoEVLModel,
)
from mlx_vlm.models.qwen3_5_moe.language import (
    LanguageModel as Qwen3_5MoEVLMLanguageModel,
)

logger = logging.getLogger(__name__)


class _MockLanguageModel:
    """Minimal stand-in for calling get_rope_index as an unbound method."""

    def __init__(self, config: Qwen3_5MoEModelConfig):
        self.config = config


class Qwen3_5MoEVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen3.5 MoE models.
    """

    def __init__(self, model_path: Path):
        super().__init__()

        self.model_cls = Qwen3_5MoEVLModel

        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=Qwen3_5MoEModelConfig,
            vision_config_class=Qwen3_5MoEVisionConfig,
            text_config_class=Qwen3_5MoETextConfig,
            vision_tower_class=Qwen3_5MoEVisionTower,
            multi_modal_projector_class=None,
            logger=logger,
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Compute input_ids and embeddings for text with images,
        then inject MRoPE position IDs into the patched text model.
        """

        input_ids, final_embeds = compute_qwen_vl_embeddings(
            addon=self,
            text_model=text_model,
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            qwen_vl_version=3,
            max_size=max_size,
        )

        # Compute and inject MRoPE position IDs for vision tokens
        if self._last_grid_thw is not None:
            mock_language_model = _MockLanguageModel(self.config)
            position_ids, rope_deltas = Qwen3_5MoEVLMLanguageModel.get_rope_index(
                mock_language_model,
                input_ids[None],
                image_grid_thw=self._last_grid_thw,
            )
            text_model.language_model.model._position_ids = position_ids
            text_model.language_model.model._rope_deltas = rope_deltas

        return input_ids, final_embeds
