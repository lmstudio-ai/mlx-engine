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
        # Older LFM2 configs keep them flat and require the custom mlx implementation.
        uses_nested_image_processor = isinstance(
            processor_config.get("image_processor"), dict
        )

        target_cls = (
            HFLfm2VlProcessor if uses_nested_image_processor else MlxLfm2VlProcessor
        )

        if not uses_nested_image_processor:
            logger.info(
                "Routing LFM2-VL processor to mlx-engine implementation (legacy flat config)"
            )
        else:
            logger.info(
                "Routing LFM2-VL processor to transformers implementation (nested config)"
            )

        return target_cls.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

    @staticmethod
    def _load_processor_config(pretrained_model_name_or_path):
        try:
            path = Path(pretrained_model_name_or_path)

            if path.is_dir():
                processor_file = path / "processor_config.json"
            else:
                processor_file = path

            if processor_file.is_file():
                return json.loads(processor_file.read_text())
        except Exception:
            logger.warning(
                "Failed to read processor_config.json for LFM2-VL; defaulting to legacy mlx processor",
                exc_info=True,
            )

        return {}
