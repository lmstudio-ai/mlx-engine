import json
from pathlib import Path
from mlx_engine.logging import log_info, log_warn


def fix_qwen2_5_vl_image_processor(model_path: Path):
    """
    Register Qwen2_5_VLImageProcessor as Qwen2VLImageProcessor in AutoImageProcessor
    This is needed because Qwen2_5_VLImageProcessor was deleted from transformers, but legacy versions of the model used this
    Ref https://github.com/Blaizzy/mlx-vlm/issues/209#issuecomment-2678113857
    Ref https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit/commit/fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b
    """
    try:
        # We are looking for a specific entry, so if any of this throws, we don't need to do anything
        with open(model_path / "preprocessor_config.json", "r") as f:
            image_processor_type = json.load(f)["image_processor_type"]
    except:  # noqa: E722
        return

    if image_processor_type != "Qwen2_5_VLImageProcessor":
        return

    log_info("Registering deprecated Qwen2_5_VLImageProcessor as Qwen2VLImageProcessor")
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLConfig,
        )
        from transformers.models.auto.processing_auto import AutoImageProcessor

        class Qwen2_5_VLImageProcessor(Qwen2VLImageProcessor):
            pass

        AutoImageProcessor.register(
            Qwen2_5_VLConfig, Qwen2_5_VLImageProcessor, exist_ok=True
        )
    except Exception as e:
        log_warn(
            f"Failed to register Qwen2_5_VLImageProcessor to AutoImageProcessor: {e}"
        )


def fix_qwen2_vl_preprocessor(model_path: Path):
    """
    Remove the `size` entry from the preprocessor_config.json file, which is broken as of transformers 5.50.0
    Ref the transformers implementation: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/e28f5d3/preprocessor_config.json
    """
    try:
        # We are looking for a specific entry, so if any of this throws, we don't need to do anything
        with open(model_path / "config.json", "r") as f:
            model_type = json.load(f)["model_type"]
        if model_type != "qwen2_vl":
            return
        with open(model_path / "preprocessor_config.json", "r") as f:
            json.load(f)["size"]
    except:  # noqa: E722
        return

    log_warn("Removing `size` entry from preprocessor_config.json")
    with open(model_path / "preprocessor_config.json", "r") as f:
        preprocessor_config = json.load(f)
    preprocessor_config.pop("size")
    with open(model_path / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f)
