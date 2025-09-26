from typing import Optional, Tuple

# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L782
VALID_KV_BITS = (2, 3, 4, 6, 8)
# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L775
VALID_KV_GROUP_SIZE = (32, 64, 128)


def get_kv_cache_quantization_params(
    kv_bits: Optional[int],
    kv_group_size: Optional[int],
    quantized_kv_start: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Validates and processes KV cache quantization parameters.

    Args:
        kv_bits: Number of bits for quantization. If None, disables quantization.
        kv_group_size: Group size for quantization. Defaults to 64 if quantization enabled.
        quantized_kv_start: Step to begin quantization. Defaults to 0 if quantization enabled.

    Returns:
        Tuple of (kv_bits, kv_group_size, quantized_kv_start), all None if quantization disabled.

    Raises:
        ValueError: If kv_bits is invalid or missing when other params are set.
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

    return kv_bits, kv_group_size, quantized_kv_start
