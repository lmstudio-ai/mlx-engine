import glob
import json
from pathlib import Path
from typing import Any, Tuple, Type

import mlx.core as mx
from mlx import nn

from mlx_vlm.utils import sanitize_weights, load_processor, get_class_predicate
from mlx_engine.logging import log_info


def load_vision_addon(
    model_path: Path,
    model_config_class: Any,
    vision_config_class: Any,
    text_config_class: Any,
    vision_tower_class: Type[nn.Module],
    multi_modal_projector_class: Type[nn.Module],
    log_prefix: str,
) -> Tuple[nn.Module, nn.Module, Any, Any]:
    """
    Load vision add-on components, configuration, and processor.

    Args:
        model_path: Path to the model directory
        model_config_class: Configuration class for the model
        vision_config_class: Configuration class for vision component
        text_config_class: Configuration class for text component
        vision_tower_class: The vision tower model class
        multi_modal_projector_class: The multi-modal projector class
        log_prefix: Prefix for logging messages

    Returns:
        Tuple containing:
            - The vision tower module
            - The multi-modal projector module
            - The model configuration
            - The processor for handling images and text
    """
    # Load and parse configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config_dict = json.loads(config_path.read_text())
    config = model_config_class.from_dict(config_dict)
    config.vision_config = vision_config_class.from_dict(config.vision_config)
    config.text_config = text_config_class.from_dict(config.text_config)

    # Create model components
    vision_tower = vision_tower_class(config.vision_config)
    multi_modal_projector = multi_modal_projector_class(config)

    # Combine components into a container module for loading weights
    class VisionComponents(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = vision_tower
            self.multi_modal_projector = multi_modal_projector

    components = VisionComponents()

    # Load processor
    processor = load_processor(model_path=model_path, add_detokenizer=True)

    # Load model weights
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(
            f"Failed to load vision add-on: {model_path} does not contain any safetensors files"
        )

    # Load and filter weights
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Filter only vision-related weights
    vision_weights = {
        k: v
        for k, v in weights.items()
        if k.startswith("vision_tower") or k.startswith("multi_modal_projector")
    }

    # Sanitize weights for vision tower
    vision_weights = sanitize_weights(
        vision_tower_class, vision_weights, config.vision_config
    )

    # Apply quantization if specified in config
    if (quantization := config_dict.get("quantization", None)) is not None:
        class_predicate = get_class_predicate(skip_vision=False, weights=vision_weights)
        nn.quantize(
            components,
            **quantization,
            class_predicate=class_predicate,
        )

    # Load weights into the model
    components.load_weights(list(vision_weights.items()))

    # Always load weights to memory here
    mx.eval(components.parameters())

    # Set model to evaluation mode
    components.eval()

    log_info(
        prefix=log_prefix,
        message=f"Vision add-on loaded successfully from {model_path}",
    )

    return vision_tower, multi_modal_projector, config, processor
