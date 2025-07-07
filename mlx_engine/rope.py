"""So...

...this isn't optimized yet. It turns out that at small sequence lengths literally just naively
applying RoPE to individual tokens is faster than doing the matrix multiplication, even with the overhead
introduced by the for loop and the theoretical optimization of the matrix multiplication. It does
begin to tip in favor of this shifting method @ larger seqlens (tested at [1,8,1000,128]) but the
overhead converting between MLX and torch still makes it slower overall than MLX-native.

- the MLX rope shift is weird anyway, but it apparently still works: TODO ask awni why
- this implementation is naive and doesn't leverage the sparsity of the RoPE matrix
- honestly it's probably easier to just use the MLX RoPE shift directly in the model
  because this allows us to not have to write custom modules for YaRN and llama3 and
  what have you, but i'll leave this here for now in case it becomes useful later
"""

import torch
import mlx.core as mx
from mlx_lm import load
import numpy as np

def mlx_rope_shift(x, shift_amount, theta=10000.0, scale=1.0, traditional=False):
    """
    MLX-compatible RoPE implementation using matrix multiplication.
    Creates a rotation matrix and applies it via matmul to shift all positions by shift_amount.
    
    Args:
        x: Input tensor of shape [bsz, n_kv_heads, seqlen, kv_head_dim]
        shift_amount: Number of positions to shift (D)
        theta: Base frequency for RoPE (default: 10000.0)
        scale: Scaling factor for frequencies (default: 1.0)
        traditional: If True, use traditional RoPE pairing (0,1), (2,3), ... 
                    If False, use MLX-style pairing (0,d/2), (1,d/2+1), ... (default: False)
    
    Returns:
        Rotated tensor of same shape as input
    """
    bsz, n_heads, seqlen, head_dim = x.shape
    device = x.device
    
    assert head_dim % 2 == 0, "Head dimension must be even"
    dim_pairs = head_dim // 2
    
    if traditional:
        # traditional RoPE: pair adjacent dimensions (0,1), (2,3), (4,5), ...
        frequencies = 1.0 / (theta ** (torch.arange(0, dim_pairs, dtype=torch.float32, device=device) * 2.0 / head_dim))
    else:
        # MLX-style RoPE: pair first half with second half (0,d/2), (1,d/2+1), ...
        frequencies = 1.0 / (theta ** (torch.arange(0, dim_pairs, dtype=torch.float32, device=device) * 2.0 / head_dim))
    
    frequencies = frequencies * scale
    angles = shift_amount * frequencies  # shape: [dim_pairs]
    cos_vals = torch.cos(angles)  # shape: [dim_pairs]
    sin_vals = torch.sin(angles)  # shape: [dim_pairs]
    
    rotation_matrix = torch.eye(head_dim, device=device, dtype=x.dtype)
    
    if traditional:
        for i in range(dim_pairs):
            even_idx = i * 2
            odd_idx = i * 2 + 1
            
            rotation_matrix[even_idx, even_idx] = cos_vals[i]
            rotation_matrix[even_idx, odd_idx] = -sin_vals[i]
            rotation_matrix[odd_idx, even_idx] = sin_vals[i]
            rotation_matrix[odd_idx, odd_idx] = cos_vals[i]
    else:
        for i in range(dim_pairs):
            first_idx = i
            second_idx = i + dim_pairs
            
            cos_val = cos_vals[i]
            sin_val = sin_vals[i]
            
            rotation_matrix[first_idx, first_idx] = cos_val
            rotation_matrix[first_idx, second_idx] = -sin_val
            rotation_matrix[second_idx, first_idx] = sin_val
            rotation_matrix[second_idx, second_idx] = cos_val
    
    rotated = x @ rotation_matrix.T
    
    return rotated


def stupid_rope(r, v, shift_by: int = 0):
    return mx.concatenate([r(v[:,:,i:i+1,:], shift_by) for i in range(v.shape[2])], axis=2)

def main():
    model, _ = load("mlx-community/Qwen3-0.6B-bf16")

    v = mx.random.normal((1, 8, 10, 128), scale=1.0, dtype=mx.float32)

    import time
    start_time = time.time()
    silly = stupid_rope(model.layers[0].self_attn.rope, v, 7)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"MLX RoPE shift took {elapsed_time:.6f} seconds")
    converted = torch.from_numpy(np.array(v))
    start_time = time.time()
    eff = mlx_rope_shift(converted, 7, theta=1000000.0, scale=1.0, traditional=False)
    end_time = time.time()
    elapsed_time2 = end_time - start_time
    print(f"Torch RoPE shift took {elapsed_time2:.6f} seconds")
    print(torch.allclose(torch.from_numpy(np.array(silly)), eff, atol=1e-5))

if __name__ == "__main__":
    main()