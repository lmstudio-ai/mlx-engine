from typing import Optional, Tuple
from mlx_lm.utils import _get_classes
from mlx_lm.models.cache import MambaCache
import logging

# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L782
VALID_KV_BITS = (2, 3, 4, 6, 8)
# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L775
VALID_KV_GROUP_SIZE = (32, 64, 128)

logger = logging.getLogger(__name__)


class _KvCacheQuantizationUnsupportedError(Exception):
    """Raised when a model doesn't support KV cache quantization."""

    def __init__(
        self,
        message="This model does not support KV Cache Quantization. Please disable and reload",
    ):
        super().__init__(message)


def get_kv_cache_quantization_params(
    kv_bits: Optional[int],
    kv_group_size: Optional[int],
    quantized_kv_start: Optional[int],
    config_json: dict,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Validates and processes KV cache quantization parameters.

    Args:
        kv_bits: Number of bits for quantization. If None, disables quantization.
        kv_group_size: Group size for quantization. Defaults to 64 if quantization enabled.
        quantized_kv_start: Step to begin quantization. Defaults to 0 if quantization enabled.
        config_json: Model config.json

    Returns:
        Tuple of (kv_bits, kv_group_size, quantized_kv_start), all None if quantization disabled.

    Raises:
        ValueError: If kv_bits is invalid or missing when other params are set.
        ValueError: If kv_bits is specified and the model arch does not support kv cache quantization
    """
    if any([kv_group_size, quantized_kv_start]) and kv_bits is None:
        raise ValueError("Enabling KV Cache Quantization requires kv_bits to be set")

    if kv_bits is None:
        return None, None, None

    # defaults taken from here:
    # https://github.com/ml-explore/mlx-examples/blob/3d793ec/llms/mlx_lm/utils.py#L352-L353
    if kv_group_size is None:
        kv_group_size = 64
    if quantized_kv_start is None:
        quantized_kv_start = 0

    if kv_bits not in VALID_KV_BITS:
        raise ValueError(f"Invalid kv_bits value. Must be one of {VALID_KV_BITS}")
    if kv_group_size not in VALID_KV_GROUP_SIZE:
        raise ValueError(
            f"Invalid kv_group_size value. Must be one of {VALID_KV_GROUP_SIZE}"
        )

    # Ensure that the model cache can be quantized
    try:
        model_cls, model_args_cls = _get_classes(config_json)
        model_args = model_args_cls.from_dict(config_json)
        model = model_cls(model_args)
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            for c in cache:
                if isinstance(c, MambaCache):
                    raise _KvCacheQuantizationUnsupportedError
    except _KvCacheQuantizationUnsupportedError as e:
        raise ValueError(str(e))
    except Exception as e:
        logger.warning(
            f"Ignoring unexpected error when checking kv cache quantization support: {e}."
        )

    return kv_bits, kv_group_size, quantized_kv_start
