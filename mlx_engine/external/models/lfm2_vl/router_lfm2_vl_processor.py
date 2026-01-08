import json
import logging
from pathlib import Path

from transformers.models.lfm2_vl.processing_lfm2_vl import (
    Lfm2VlProcessor as HFLfm2VlProcessor,
)

from .processing_lfm2_vl import Lfm2VlProcessor as MlxLfm2VlProcessor

logger = logging.getLogger(__name__)


class Lfm2VlProcessor:
    """
    Minimal shim that routes processor instantiation to the correct implementation
    based on the saved processor_config.json shape.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor_config = cls._load_processor_config(pretrained_model_name_or_path)

        # Newer LFM2.5-style configs nest vision settings under "image_processor".
        # Older LFM2 configs keep them flat and require the custom mlx-engine implementation.
        uses_nested_image_processor = isinstance(
            processor_config.get("image_processor"), dict
        )

        if uses_nested_image_processor:
            target_cls = HFLfm2VlProcessor
            logger.info(
                "Routing LFM2-VL processor to transformers implementation (nested config)"
            )
        else:
            target_cls = MlxLfm2VlProcessor
            logger.info(
                "Routing LFM2-VL processor to mlx-engine implementation (legacy flat config)"
            )

        return target_cls.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

    @staticmethod
    def _load_processor_config(pretrained_model_name_or_path):
        path = Path(pretrained_model_name_or_path)

        if not path.is_dir():
            raise ValueError(
                f"LFM2-VL processor requires a local directory path, got: {pretrained_model_name_or_path}"
            )

        processor_file = path / "processor_config.json"
        if not processor_file.is_file():
            raise ValueError(
                f"processor_config.json not found in {pretrained_model_name_or_path}"
            )

        return json.loads(processor_file.read_text())
