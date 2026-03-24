import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import compute_qwen_vl_embeddings

from mlx_vlm.models.qwen3_5 import (
    VisionModel as Qwen3_5VisionTower,
    ModelConfig as Qwen3_5ModelConfig,
    VisionConfig as Qwen3_5VisionConfig,
    TextConfig as Qwen3_5TextConfig,
    Model as Qwen3_5VLModel,
)
from mlx_vlm.models.qwen3_5.language import LanguageModel as Qwen3_5VLMLanguageModel

logger = logging.getLogger(__name__)


class _MockLanguageModel:
    """Minimal stand-in for calling get_rope_index as an unbound method."""

    def __init__(self, config: Qwen3_5ModelConfig):
        self.config = config


class Qwen3_5VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen3.5 Dense models.
    """

    def __init__(self, model_path: Path):
        super().__init__()
        self._init_common(
            model_path=model_path,
            model_cls=Qwen3_5VLModel,
            language_model_cls=Qwen3_5VLMLanguageModel,
            model_config_class=Qwen3_5ModelConfig,
            vision_config_class=Qwen3_5VisionConfig,
            text_config_class=Qwen3_5TextConfig,
            vision_tower_class=Qwen3_5VisionTower,
            addon_logger=logger,
        )

    def _init_common(
        self,
        model_path,
        model_cls,
        language_model_cls,
        model_config_class,
        vision_config_class,
        text_config_class,
        vision_tower_class,
        addon_logger,
    ):
        """Shared initialization for dense and MoE variants."""
        self._last_grid_thw = None
        self.model_cls = model_cls
        self._language_model_cls = language_model_cls
        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=model_config_class,
            vision_config_class=vision_config_class,
            text_config_class=text_config_class,
            vision_tower_class=vision_tower_class,
            multi_modal_projector_class=None,
            logger=addon_logger,
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
            position_ids, rope_deltas = self._language_model_cls.get_rope_index(
                mock_language_model,
                input_ids[None],
                image_grid_thw=self._last_grid_thw,
            )
            text_model.language_model.model._position_ids = position_ids
            text_model.language_model.model._rope_deltas = rope_deltas

        return input_ids, final_embeds
