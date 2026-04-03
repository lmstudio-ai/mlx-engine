import glob
import json
from pathlib import Path
from typing import Any, Tuple, Type
import mlx.core as mx
from mlx import nn
from mlx_vlm.utils import sanitize_weights, load_processor, skip_multimodal_module
import logging


def load_and_parse_config(
    model_path: Path,
    model_config_class: Any,
    vision_config_class: Any,
    text_config_class: Any,
) -> Tuple[Any, dict]:
    """
    Load and parse vision model configuration from config.json.

    Args:
        model_path: Path to the model directory
        model_config_class: Configuration class for the model
        vision_config_class: Configuration class for vision component
        text_config_class: Configuration class for text component

    Returns:
        Tuple containing:
            - The fully initialized config object
            - The raw config dictionary (needed for quantization later)
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config_dict = json.loads(config_path.read_text())
    config = model_config_class.from_dict(config_dict)
    config.vision_config = vision_config_class.from_dict(config.vision_config)
    config.text_config = text_config_class.from_dict(config.text_config)

    # hack for lfm2_vl, which uses a `vision_feature_layer` to reduce the number of actual layers
    # https://github.com/Blaizzy/mlx-vlm/blob/f02d63e8f5b521e8c75f129a63d2660efd132693/mlx_vlm/models/lfm2_vl/lfm2_vl.py#L98-L101
    if (
        hasattr(config.text_config, "model_type")
        and "lfm2" in config.text_config.model_type
    ):
        vision_feature_layer = config_dict.get("vision_feature_layer", -1)
        if vision_feature_layer != -1:
            config.vision_config.num_hidden_layers += vision_feature_layer + 1
            config_dict["vision_config"]["num_hidden_layers"] = (
                config.vision_config.num_hidden_layers
            )

    return config, config_dict


class VisionComponents(nn.Module):
    def __init__(
        self, vision_tower: nn.Module, multi_modal_projector: nn.Module | None = None
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector


def create_vision_components(
    config: Any,
    vision_tower_class: Type[nn.Module],
    multi_modal_projector_class: Type[nn.Module] | None,
) -> VisionComponents:
    """
    Create vision model components and wrap them in a container module.

    Args:
        config: The fully initialized config object
        vision_tower_class: The vision tower model class
        multi_modal_projector_class: The multi-modal projector class

    Returns:
        The container module with both components
    """
    components = VisionComponents(
        vision_tower_class(config.vision_config),
        multi_modal_projector_class(config) if multi_modal_projector_class else None,
    )
    return components


def load_and_filter_weights(
    model_path: Path,
    components: nn.Module,
) -> dict:
    """
    Load model weights from safetensors files and filter for vision-related weights.

    Args:
        model_path: Path to the model directory
        components: The vision components container module

    Returns:
        Dictionary containing filtered vision-related weights
    """
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
        if any(k.startswith(name) for name in components.children().keys())
    }

    return vision_weights


def maybe_apply_quantization(
    components: nn.Module,
    config_dict: dict,
    vision_weights: dict,
) -> None:
    """
    Apply quantization to vision components if specified in config.

    Args:
        components: The vision components container module
        config_dict: Raw config dictionary containing quantization settings
        vision_weights: The vision-related weights dictionary
    """
    # Apply quantization if specified in config
    if (quantization := config_dict.get("quantization", None)) is not None:
        # Copied from mlx_vlm/utils.py at commit
        # 65ecc837f24d0f8b138f300c7efef87f00fba74d
        skip_vision = config_dict.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            # Always skip vision and audio models
            if skip_multimodal_module(p) and skip_vision:
                return False
            # Handle custom per layer quantizations
            if p in config_dict["quantization"]:
                return config_dict["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in vision_weights

        quantize_kwargs = {}
        if "bits" in quantization:
            quantize_kwargs["bits"] = quantization["bits"]
        if "group_size" in quantization:
            quantize_kwargs["group_size"] = quantization["group_size"]
        if "mode" in quantization:
            quantize_kwargs["mode"] = quantization["mode"]
        nn.quantize(
            components,
            class_predicate=get_class_predicate,
            **quantize_kwargs,
        )


def prepare_components(
    components: nn.Module,
    vision_weights: dict,
) -> None:
    """
    Prepare vision components by loading weights and setting to evaluation mode.

    Args:
        components: The vision components container module
        vision_weights: The vision-related weights dictionary
    """
    # Load weights into the model
    components.load_weights(list(vision_weights.items()))

    # Always load weights to memory here
    mx.eval(components.parameters())

    # Set model to evaluation mode
    components.eval()


def load_vision_addon(
    model_path: Path,
    model_config_class: Any,
    vision_config_class: Any,
    text_config_class: Any,
    vision_tower_class: Type[nn.Module],
    multi_modal_projector_class: Type[nn.Module] | None,
    logger: logging.Logger,
    processor_kwargs: dict | None = None,
) -> Tuple[nn.Module, nn.Module | None, Any, Any]:
    """
    Load vision add-on components, configuration, and processor.

    Args:
        model_path: Path to the model directory
        model_config_class: Configuration class for the model
        vision_config_class: Configuration class for vision component
        text_config_class: Configuration class for text component
        vision_tower_class: The vision tower model class
        multi_modal_projector_class: The multi-modal projector class
        logger: logging.Logger

    Returns:
        Tuple containing:
            - The vision tower module
            - The multi-modal projector module
            - The model configuration
            - The processor for handling images and text
    """
    # Load and parse configuration
    config, config_dict = load_and_parse_config(
        model_path, model_config_class, vision_config_class, text_config_class
    )

    # Create model components
    components = create_vision_components(
        config,
        vision_tower_class,
        multi_modal_projector_class,
    )

    # Load processor
    processor = load_processor(
        model_path=model_path,
        add_detokenizer=True,
        **(processor_kwargs or {}),
    )

    # Load and filter weights
    vision_weights = load_and_filter_weights(model_path, components)

    # Sanitize weights for vision tower
    vision_weights = sanitize_weights(
        components.vision_tower.__class__, vision_weights, config.vision_config
    )

    # Apply quantization if specified in config
    maybe_apply_quantization(components, config_dict, vision_weights)

    # Prepare components (load weights and set to eval mode)
    prepare_components(components, vision_weights)

    logger.info(
        f"Vision add-on loaded successfully from {model_path}",
    )

    return components.vision_tower, components.multi_modal_projector, config, processor
