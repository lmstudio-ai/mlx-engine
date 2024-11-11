import numpy as np
import torch
import mlx.core as mx
import time
from typing import Optional
import random


def set_seed(seed: Optional[int]) -> None:
    """
    Set the seed for all random number generators used in mlx-engine.

    Args:
        seed: The seed to use. If None, the seed will be set to the current nanosecond timestamp.
    """
    if seed is None:
        # Get nanosecond timestamp and use it as seed
        seed = int(time.time_ns()) & (2**32 - 1)  # Ensure seed fits in 32 bits

    # For MLX and MLX_LM
    mx.random.seed(seed)

    # MLX_VLM depends on numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Just in case
    random.seed(seed)
