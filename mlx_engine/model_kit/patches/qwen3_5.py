"""
Qwen3.5 MRoPE patch using derive and override pattern.

The mlx-lm Qwen3.5 model uses standard RoPE (via Qwen3NextAttention), which works
for text-only generation. However, Qwen3.5 requires Multimodal RoPE (MRoPE) with
3D position IDs for vision tasks.

This patch replaces the attention module with mlx-vlm's MRoPE-capable version and
adds position_ids threading through the model. For text-only (position_ids=None),
the MRoPE attention falls back to sequential 3D positions, which is mathematically
equivalent to standard RoPE.

Implementation inspired by:
  mlx-lm @ 564281f79328df07c4997b3a6ca00bd929381287
  mlx-vlm @ 822a843941ea35ddee2849fb0633a80eac1d1d94
"""

from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models.qwen3_5 import (
    DecoderLayer,
    Qwen3_5TextModel,
)
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention


class PatchedDecoderLayer(DecoderLayer):
    """
    DecoderLayer that uses MRoPE-capable attention and accepts position_ids.

    Replaces the standard Qwen3NextAttention with Qwen3_5Attention from mlx-vlm,
    which supports interleaved multimodal RoPE with 3D position IDs. GatedDeltaNet
    layers and MLP/SparseMoeBlock are untouched from the original.
    """

    def __init__(self, args, layer_idx):
        super().__init__(args, layer_idx)
        if not self.is_linear:
            self.self_attn = Qwen3_5Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class PatchedQwen3_5TextModel(Qwen3_5TextModel):
    """
    Qwen3_5TextModel with MRoPE position state management.

    Adds _position_ids and _rope_deltas attributes that can be set externally
    (by the vision add-on) before generation. During forward passes, computes
    the appropriate position_ids from this state and threads them to decoder layers.

    Position state logic (ported from mlx_vlm LanguageModel.__call__):
    - Both None: text-only path, passes position_ids=None (attention uses cache offset)
    - _position_ids set: prefill, slices stored positions by cache offset
    - Only _rope_deltas set: autoregressive generation, computes from delta
    """

    def __init__(self, args):
        super().__init__(args)
        self._position_ids = None
        self._rope_deltas = None

    def reset_mrope_state(self):
        """
        Reset MRoPE position state.

        Called by the vision add-on's clear_prediction_state before every
        prediction. For vision requests, compute_embeddings sets fresh
        state immediately after.
        """
        self._position_ids = None
        self._rope_deltas = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        position_ids = self._compute_position_ids(inputs, cache)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=layer_cache, position_ids=position_ids
            )

        return self.norm(hidden_states)

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """
        Compute position_ids for the current forward pass from stored state.

        Ported from mlx_vlm.models.qwen3_5.language.LanguageModel.__call__ lines 594-651.

        Branching logic:
        - Both state attrs None: text-only, return None (attention uses cache offset)
        - cache_offset == 0 and _position_ids set: first prefill chunk, slice stored positions
        - cache_offset > 0 and _rope_deltas set: subsequent chunks / autoregressive, compute
          sequential positions from rope_deltas
        """
        if self._position_ids is None and self._rope_deltas is None:
            return None

        cache_offset = 0
        if cache is not None and cache[self.fa_idx] is not None:
            offset = cache[self.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()

        # Use stored position_ids for the first prefill chunk (cache_offset == 0),
        # or when rope_deltas hasn't been computed yet, or when there's no cache.
        use_stored_positions = (
            (cache is not None and cache[self.fa_idx] is not None and cache_offset == 0)
            or self._rope_deltas is None
            or cache is None
        )

        if use_stored_positions and self._position_ids is not None:
            seq_length = inputs.shape[1]
            return self._position_ids[:, :, cache_offset : cache_offset + seq_length]

        # Subsequent prefill chunks and autoregressive: compute from rope_deltas
        batch_size, seq_length = inputs.shape
        delta = mx.array(cache_offset + self._rope_deltas if cache is not None else 0)
        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if delta.ndim == 0:
            delta = mx.expand_dims(delta, axis=0)
        if delta.shape[0] < batch_size:
            delta = mx.tile(delta, (batch_size, 1))
        else:
            delta = delta[:batch_size]

        position_ids = mx.add(position_ids, delta)[None, ...]
        return mx.broadcast_to(position_ids, (3, batch_size, seq_length))


def apply_patches():
    """
    Apply Qwen3.5 MRoPE patches by replacing classes in the mlx_lm module.
    """
    import mlx_lm.models.qwen3_5

    mlx_lm.models.qwen3_5.DecoderLayer = PatchedDecoderLayer
    mlx_lm.models.qwen3_5.Qwen3_5TextModel = PatchedQwen3_5TextModel
