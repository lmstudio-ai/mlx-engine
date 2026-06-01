"""TurboQuant: PolarQuant + Walsh-Hadamard rotation for KV cache compression.

Port of TurboQuant+ (github.com/TheTom/turboquant_plus) to MLX.
Implements the PolarQuant algorithm (AISTATS 2026) for near-lossless KV cache
compression: norm extraction -> WHT rotation -> Lloyd-Max scalar quantization.

For the full TurboQuant pipeline (including QJL residual stage), see the
original NumPy reference at 0xSero/turboquant.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fast_walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Fast Walsh-Hadamard Transform using MLX ops, O(n log n).

    Args:
        x: Input array, 1D or 2D (batch, n). Last dim must be power of 2.

    Returns:
        Transformed array, normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    single = x.ndim == 1
    if single:
        x = x[None, :]

    batch = x.shape[0]
    h = 1
    while h < n:
        x = x.reshape(batch, n // (h * 2), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = mx.concatenate([a + b, a - b], axis=-1)
        x = x.reshape(batch, n)
        h *= 2

    x = x / math.sqrt(n)
    return x[0] if single else x


def make_wh_rotation(d: int, seed: int = 42) -> Tuple[mx.array, mx.array, int]:
    """Create a fast Walsh-Hadamard rotation: D1 @ H @ D2.

    Args:
        d: Input dimension.
        seed: Random seed for sign flips.

    Returns:
        Tuple of (signs1, signs2, padded_d) where signs are +1/-1 arrays
        and padded_d is the next power of 2 >= d.
    """
    rng = np.random.default_rng(seed)
    padded_d = _next_power_of_2(d)
    signs1 = mx.array(rng.choice([-1.0, 1.0], size=padded_d))
    signs2 = mx.array(rng.choice([-1.0, 1.0], size=padded_d))
    return signs1, signs2, padded_d


def apply_wh_rotation(x: mx.array, signs1: mx.array, signs2: mx.array, padded_d: int) -> mx.array:
    """Apply WHT rotation: x_rot = D2 @ H @ D1 @ pad(x).

    Args:
        x: Input vector(s), shape (d,) or (batch, d).
        signs1, signs2: Sign arrays from make_wh_rotation.
        padded_d: Padded dimension.

    Returns:
        Rotated vector(s), shape (d,) or (batch, d).
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]

    batch, d = x.shape
    padded = mx.zeros((batch, padded_d))
    padded[:, :d] = x
    padded = padded * signs1[None, :]
    padded = fast_walsh_hadamard_transform(padded)
    padded = padded * signs2[None, :]
    result = padded[:, :d]
    return result[0] if single else result


def apply_wh_rotation_transpose(y: mx.array, signs1: mx.array, signs2: mx.array, padded_d: int) -> mx.array:
    """Apply inverse WHT rotation: y_unrot = D1 @ H @ D2 @ pad(y).

    Since D1, D2, H are their own transpose (symmetric), the inverse
    applies them in reverse order.
    """
    single = y.ndim == 1
    if single:
        y = y[None, :]

    batch, d = y.shape
    padded = mx.zeros((batch, padded_d))
    padded[:, :d] = y
    padded = padded * signs2[None, :]
    padded = fast_walsh_hadamard_transform(padded)
    padded = padded * signs1[None, :]
    result = padded[:, :d]
    return result[0] if single else result


def lloyd_max_centroids(bit_width: int, d: int) -> mx.array:
    """Compute optimal MSE centroids for N(0, 1/d) distribution.

    For 1-bit and 2-bit, uses closed-form solutions.
    For 3+ bits, uses precomputed values from scipy-based Lloyd's algorithm.

    Args:
        bit_width: Number of bits (1, 2, 3, 4).
        d: Vector dimension (affects scale).

    Returns:
        Sorted centroid array, shape (2^bit_width,).
    """
    n_centroids = 1 << bit_width
    sigma = 1.0 / math.sqrt(d)

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return mx.array([-c, c])

    if bit_width == 2:
        return mx.array([-1.51, -0.453, 0.453, 1.51]) / math.sqrt(d)

    # Precomputed values for N(0,1) from scipy.stats based Lloyd iterations
    # Scale by sigma for the target distribution
    gaussian_centroids = {
        3: mx.array([-2.205, -1.259, -0.381, 0.381, 1.259, 2.205]),
        4: mx.array([-2.788, -1.880, -1.087, -0.363, 0.363, 1.087, 1.880, 2.788]),
    }
    if bit_width in gaussian_centroids:
        return gaussian_centroids[bit_width] * sigma

    raise ValueError(f"Unsupported bit_width: {bit_width}")


def quantize_polar(x: mx.array, centroids: mx.array, signs1: mx.array, signs2: mx.array, padded_d: int) -> Tuple[mx.array, mx.array]:
    """PolarQuant: quantize vectors using PolarQuant algorithm.

    Args:
        x: Input vectors, shape (batch, d).
        centroids: Precomputed centroids from lloyd_max_centroids.
        signs1, signs2, padded_d: WHT rotation parameters.

    Returns:
        Tuple of (indices, norms) where indices shape is (batch, d) of uint8
        and norms shape is (batch,) of float32.
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]

    batch, d = x.shape

    # 1. Extract norms and normalize
    norms = mx.linalg.norm(x, axis=1)
    safe_norms = mx.where(norms > 0, norms, 1.0)
    x_normalized = x / safe_norms[:, None]

    # 2. Apply WHT rotation
    y = apply_wh_rotation(x_normalized, signs1, signs2, padded_d)

    # 3. Quantize each coordinate to nearest centroid
    # Compute boundaries between centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    # For each value, find which boundary interval it falls into
    # This uses broadcasting: y (batch, d, 1) - boundaries (1, n-1) -> (batch, d, n-1)
    # Count how many boundaries are less than each value
    y_expanded = y[:, :, None]
    boundaries_expanded = boundaries[None, None, :]
    indices = mx.sum(y_expanded > boundaries_expanded, axis=-1).astype(mx.uint8)

    if single:
        indices = indices[0]
        norms = norms[0]

    return indices, norms


def dequantize_polar(indices: mx.array, norms: mx.array, centroids: mx.array,
                     signs1: mx.array, signs2: mx.array, padded_d: int,
                     norm_correction: bool = True) -> mx.array:
    """Dequantize PolarQuant indices back to vectors.

    Args:
        indices: Shape (batch, d) of uint8 centroid indices.
        norms: Shape (batch,) of L2 norms.
        centroids: Precomputed centroids.
        signs1, signs2, padded_d: WHT rotation parameters.
        norm_correction: Apply norm correction step.

    Returns:
        Reconstructed vectors, same shape as original (batch, d).
    """
    single = indices.ndim == 1
    if single:
        indices = indices[None, :]
        norms = norms[None]

    # 1. Look up centroids
    y_hat = centroids[indices.astype(mx.int32)]

    # 2. Norm correction (renormalize y_hat to unit norm)
    if norm_correction:
        y_hat_norms = mx.linalg.norm(y_hat, axis=1, keepdims=True)
        y_hat_norms = mx.where(y_hat_norms > 1e-10, y_hat_norms, 1.0)
        y_hat = y_hat / y_hat_norms

    # 3. Inverse rotation
    x_hat_unit = apply_wh_rotation_transpose(y_hat, signs1, signs2, padded_d)

    # 4. Rescale by original norms
    x_hat = x_hat_unit * norms[:, None]

    if single:
        x_hat = x_hat[0]

    return x_hat


class TurboQuantKVCache:
    """MLX-compatible KV cache with TurboQuant PolarQuant compression.

    Wraps a standard KVCache and compresses keys/values on write using
    PolarQuant. Decompresses on read so attention computations work normally.

    This is a drop-in replacement for KVCache in the MLX pipeline.
    """

    def __init__(self, head_dim: int, k_bits: int = 3, v_bits: int = 3,
                 seed: int = 42, norm_correction: bool = True):
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits

        self.signs1_k, self.signs2_k, self.padded_d_k = make_wh_rotation(head_dim, seed)
        self.signs1_v, self.signs2_v, self.padded_d_v = make_wh_rotation(head_dim, seed + 500)

        self.centroids_k = lloyd_max_centroids(k_bits, head_dim)
        self.centroids_v = lloyd_max_centroids(v_bits, head_dim)

        self.norm_correction = norm_correction

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

        # Metadata for the compressed cache
        self.k_quantized: Optional[mx.array] = None  # quantized key indices
        self.k_norms: Optional[mx.array] = None
        self.v_quantized: Optional[mx.array] = None
        self.v_norms: Optional[mx.array] = None

    def quantize_kv(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize keys and values using PolarQuant.

        Args:
            keys: Shape (B, n_kv_heads, seq_len, head_dim).
            values: Same shape as keys.

        Returns:
            Tuple of (quantized_keys, quantized_values) where
            quantized_keys/v are indices into the codebook.
        """
        B, n_kv_heads, seq_len, head_dim = keys.shape

        # Reshape to (B*n_kv_heads*seq_len, head_dim) for batch quantization
        k_flat = keys.reshape(-1, head_dim)
        v_flat = values.reshape(-1, head_dim)

        k_idx, k_norm = quantize_polar(k_flat, self.centroids_k,
                                        self.signs1_k, self.signs2_k, self.padded_d_k)
        v_idx, v_norm = quantize_polar(v_flat, self.centroids_v,
                                        self.signs1_v, self.signs2_v, self.padded_d_v)

        # Reshape back
        k_idx = k_idx.reshape(B, n_kv_heads, seq_len, head_dim)
        v_idx = v_idx.reshape(B, n_kv_heads, seq_len, head_dim)
        k_norm = k_norm.reshape(B, n_kv_heads, seq_len)
        v_norm = v_norm.reshape(B, n_kv_heads, seq_len)

        return k_idx, v_idx, k_norm, v_norm

    def dequantize_kv(self, k_idx: mx.array, v_idx: mx.array,
                      k_norm: mx.array, v_norm: mx.array) -> Tuple[mx.array, mx.array]:
        """Dequantize compressed keys and values.

        Args:
            k_idx: Shape (B, n_kv_heads, seq_len, head_dim).
            v_idx: Same shape.
            k_norm: Shape (B, n_kv_heads, seq_len).
            v_norm: Same shape.

        Returns:
            Tuple of (keys, values) in fp32.
        """
        B, n_kv_heads, seq_len, head_dim = k_idx.shape

        k_flat = k_idx.reshape(-1, head_dim)
        v_flat = v_idx.reshape(-1, head_dim)
        kn_flat = k_norm.reshape(-1)
        vn_flat = v_norm.reshape(-1)

        k_deq = dequantize_polar(k_flat, kn_flat, self.centroids_k,
                                  self.signs1_k, self.signs2_k, self.padded_d_k,
                                  self.norm_correction)
        v_deq = dequantize_polar(v_flat, vn_flat, self.centroids_v,
                                  self.signs1_v, self.signs2_v, self.padded_d_v,
                                  self.norm_correction)

        k_deq = k_deq.reshape(B, n_kv_heads, seq_len, head_dim)
        v_deq = v_deq.reshape(B, n_kv_heads, seq_len, head_dim)

        return k_deq, v_deq

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache with new keys/values and return the full cache.

        Compresses incoming KV pairs, appends to compressed store,
        and returns the dequantized full cache for attention computation.

        Args:
            keys: New key tensor, shape (B, n_kv_heads, seq_len, head_dim).
            values: New value tensor, same shape.

        Returns:
            Dequantized full key and value tensors.
        """
        B, n_kv_heads, new_len, head_dim = keys.shape

        # Quantize the new keys/values
        k_idx, v_idx, k_norm, v_norm = self.quantize_kv(keys, values)

        # Append to compressed storage
        if self.k_quantized is None:
            self.k_quantized = k_idx
            self.k_norms = k_norm
            self.v_quantized = v_idx
            self.v_norms = v_norm
        else:
            self.k_quantized = mx.concatenate([self.k_quantized, k_idx], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_norm], axis=2)
            self.v_quantized = mx.concatenate([self.v_quantized, v_idx], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_norm], axis=2)

        self.offset += new_len

        # Dequantize full cache for attention
        return self.dequantize_kv(
            self.k_quantized, self.v_quantized,
            self.k_norms, self.v_norms,
        )

    @property
    def state(self) -> Tuple[mx.array, mx.array]:
        """Return dequantized state for compatibility with MLX attention layers."""
        if self.k_quantized is None:
            return None, None
        return self.dequantize_kv(
            self.k_quantized, self.v_quantized,
            self.k_norms, self.v_norms,
        )

    @state.setter
    def state(self, v: Tuple[mx.array, mx.array]) -> None:
        """Re-quantize from full-precision state (for checkpoint restore)."""
        keys, values = v
        self.offset = keys.shape[2]
        k_idx, v_idx, k_norm, v_norm = self.quantize_kv(keys, values)
        self.k_quantized = k_idx
        self.k_norms = k_norm
        self.v_quantized = v_idx
        self.v_norms = v_norm

    def size(self) -> int:
        """Number of cached tokens."""
        return self.offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim cache by n tokens."""
        n = min(self.offset, n)
        if self.k_quantized is not None:
            self.k_quantized = self.k_quantized[..., :-n, :]
            self.k_norms = self.k_norms[..., :-n]
            self.v_quantized = self.v_quantized[..., :-n, :]
            self.v_norms = self.v_norms[..., :-n]
        self.offset -= n
        return n

    def empty(self) -> bool:
        return self.k_quantized is None

    @property
    def nbytes(self) -> int:
        if self.k_quantized is None:
            return 0
        # Compressed size: indices (uint8) + norms (float32)
        n_elements = self.k_quantized.size
        return n_elements * (1 + 4)  # uint8 + float32 per element
