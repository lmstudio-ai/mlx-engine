import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import compute_qwen_vl_embeddings

from mlx_vlm.models.qwen3_vl_moe import (
    VisionModel as Qwen3_VL_MoEVisionTower,
    ModelConfig as Qwen3_VL_MoEModelConfig,
    VisionConfig as Qwen3_VL_MoEVisionConfig,
    TextConfig as Qwen3_VL_MoETextConfig,
    Model as Qwen3_VL_MoEModel,
)

logger = logging.getLogger(__name__)


class Qwen3_VL_MoEVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen3-VL MoE models.
    """

    def __init__(self, model_path: Path):
        """Initialize Qwen3_VL_MoEVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Store the model class for use in compute_embeddings
        self.model_cls = Qwen3_VL_MoEModel

        # Load vision components
        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=Qwen3_VL_MoEModelConfig,
            vision_config_class=Qwen3_VL_MoEVisionConfig,
            text_config_class=Qwen3_VL_MoETextConfig,
            vision_tower_class=Qwen3_VL_MoEVisionTower,
            multi_modal_projector_class=None,
            logger=logger,
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
    ) -> tuple[mx.array, mx.array]:
        """
        Compute input_ids and embeddings for text with images.
        """

        return compute_qwen_vl_embeddings(
            addon=self,
            text_model=text_model,
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            qwen_vl_version=3,
        )
