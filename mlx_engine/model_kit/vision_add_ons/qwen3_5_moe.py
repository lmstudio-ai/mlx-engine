import logging
from pathlib import Path

from mlx_engine.model_kit.vision_add_ons.qwen3_5 import Qwen3_5VisionAddOn

from mlx_vlm.models.qwen3_5_moe import (
    VisionModel as Qwen3_5MoEVisionTower,
    ModelConfig as Qwen3_5MoEModelConfig,
    VisionConfig as Qwen3_5MoEVisionConfig,
    TextConfig as Qwen3_5MoETextConfig,
    Model as Qwen3_5MoEVLModel,
)

logger = logging.getLogger(__name__)


class Qwen3_5MoEVisionAddOn(Qwen3_5VisionAddOn):
    """
    Vision add-on for Qwen3.5 MoE models.
    Subclasses the dense variant, overriding only __init__ with MoE-specific imports.
    """

    def __init__(self, model_path: Path):
        # Skip Qwen3_5VisionAddOn.__init__ and call BaseVisionAddOn.__init__ directly,
        # since we need entirely different config/model classes.
        super(Qwen3_5VisionAddOn, self).__init__()
        self._init_common(
            model_path=model_path,
            model_cls=Qwen3_5MoEVLModel,
            model_config_class=Qwen3_5MoEModelConfig,
            vision_config_class=Qwen3_5MoEVisionConfig,
            text_config_class=Qwen3_5MoETextConfig,
            vision_tower_class=Qwen3_5MoEVisionTower,
            addon_logger=logger,
        )
