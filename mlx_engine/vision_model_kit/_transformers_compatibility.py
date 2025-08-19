import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def fix_qwen2_5_vl_image_processor(model_path: Path):
    """
    Update the `image_processor_type` in the preprocessor_config.json file to Qwen2VLImageProcessor
    Ref https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit/commit/fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b
    """
    try:
        # We are looking for a specific entry, so if any of this throws, we don't need to do anything
        with open(model_path / "preprocessor_config.json", "r") as f:
            image_processor_type = json.load(f)["image_processor_type"]
        with open(model_path / "config.json", "r") as f:
            model_type = json.load(f)["model_type"]
    except:  # noqa: E722
        return

    if not (
        image_processor_type == "Qwen2_5_VLImageProcessor"
        and model_type == "qwen2_5_vl"
    ):
        return

    # Replace image_processor_type with Qwen2VLImageProcessor
    logger.warning(
        "Replacing `image_processor_type` with Qwen2VLImageProcessor in preprocessor_config.json"
    )
    with open(model_path / "preprocessor_config.json", "r") as f:
        preprocessor_config = json.load(f)
    preprocessor_config["image_processor_type"] = "Qwen2VLImageProcessor"
    with open(model_path / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f)


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

    logger.warning("Removing `size` entry from preprocessor_config.json")
    with open(model_path / "preprocessor_config.json", "r") as f:
        preprocessor_config = json.load(f)
    preprocessor_config.pop("size")
    with open(model_path / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f)
