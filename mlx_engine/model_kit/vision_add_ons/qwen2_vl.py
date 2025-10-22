import logging
import json
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import compute_qwen_vl_embeddings

from mlx_vlm.models.qwen2_5_vl import (
    VisionModel as Qwen25VLVisionTower,
    ModelConfig as Qwen25VLModelConfig,
    VisionConfig as Qwen25VLVisionConfig,
    TextConfig as Qwen25VLTextConfig,
    Model as Qwen25VLModel,
)
from mlx_vlm.models.qwen2_vl import (
    VisionModel as Qwen2VLVisionTower,
    ModelConfig as Qwen2VLModelConfig,
    VisionConfig as Qwen2VLVisionConfig,
    TextConfig as Qwen2VLTextConfig,
    Model as Qwen2VLModel,
)

logger = logging.getLogger(__name__)


class Qwen2_VLVisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen2-VL and Qwen2.5-VL models.
    """

    def __init__(self, model_path: Path):
        """Initialize Qwen2_VLVisionAddOn with vision components loaded from the given path."""
        super().__init__()

        # Determine model type from config to select appropriate classes
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            model_type = config_dict.get("model_type")

        # Import appropriate classes based on model type
        if model_type == "qwen2_5_vl":
            vision_tower_cls = Qwen25VLVisionTower
            model_config_cls = Qwen25VLModelConfig
            vision_config_cls = Qwen25VLVisionConfig
            text_config_cls = Qwen25VLTextConfig
            model_cls = Qwen25VLModel
        else:  # Default to qwen2_vl
            vision_tower_cls = Qwen2VLVisionTower
            model_config_cls = Qwen2VLModelConfig
            vision_config_cls = Qwen2VLVisionConfig
            text_config_cls = Qwen2VLTextConfig
            model_cls = Qwen2VLModel

        # Store the model class for use in compute_embeddings
        self.model_cls = model_cls

        # Load vision components
        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=model_config_cls,
            vision_config_class=vision_config_cls,
            text_config_class=text_config_cls,
            vision_tower_class=vision_tower_cls,
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
        Compute input_ids and embeddings for text with images.
        """

        return compute_qwen_vl_embeddings(
            addon=self,
            text_model=text_model,
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            qwen_vl_version=2,
            max_size=max_size,
        )
