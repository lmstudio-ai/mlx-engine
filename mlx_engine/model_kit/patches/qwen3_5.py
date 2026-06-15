"""
Qwen3.5 MRoPE patch using derive and override pattern.

The mlx-lm Qwen3.5 model uses standard RoPE (via Qwen3NextAttention), which works
for text-only generation. However, Qwen3.5 requires Multimodal RoPE (MRoPE) with
3D position IDs for vision tasks.

This patch adds a dual code path to each decoder layer, selected by position_ids:

- Text-only (position_ids=None): uses the original mlx-lm modules (Qwen3NextAttention
  with nn.RoPE, GatedDeltaNet with _precise_swiglu) — bit-identical to unpatched mlx-lm.
- Vision (position_ids provided): mirrors mlx-vlm's computation (MRoPE attention,
  GatedDeltaNet with the shared norm/output path) — bit-identical to native mlx-vlm.

Both paths read weights from the same modules; no weight duplication.

Reference implementations:
  https://github.com/ml-explore/mlx-lm/blob/aa4f880/mlx_lm/models/qwen3_5.py#L86-L206
  https://github.com/Blaizzy/mlx-vlm/blob/58e2435/mlx_vlm/models/qwen3_5/language.py#L92-L356
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import (
    DecoderLayer,
    Qwen3_5TextModel,
)
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.qwen3_next import Qwen3NextAttention
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.qwen3_5 import language as vlm_qwen3_5_language
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel as VlmQwen3_5LanguageModel,
    Qwen3_5Attention as VlmQwen3_5Attention,
    Qwen3_5GatedDeltaNet as VlmQwen3_5GatedDeltaNet,
    Qwen3_5RotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)

# Stable aliases to the pristine mlx-lm classes captured before apply_patches()
# mutates mlx_lm.models.qwen3_5 in place.
OriginalDecoderLayer = DecoderLayer
OriginalQwen3_5TextModel = Qwen3_5TextModel
OriginalVlmQwen3_5AttentionInit = VlmQwen3_5Attention.__init__
OriginalVlmQwen3_5AttentionCall = VlmQwen3_5Attention.__call__
OriginalVlmQwen3_5GatedDeltaNetCall = VlmQwen3_5GatedDeltaNet.__call__
OriginalVlmQwen3_5LanguageModelCall = VlmQwen3_5LanguageModel.__call__
OriginalVlmQwen3_5IsSingleRowBatchCache = (
    vlm_qwen3_5_language._is_single_row_batch_cache
)
OriginalVlmQwen3_5RaggedDecodeAttention = (
    vlm_qwen3_5_language._qwen3_5_ragged_decode_attention
)


def _patched_vlm_qwen3_5_ragged_decode_attention(*args, **kwargs):
    return None


def _vlm_qwen3_5_gated_delta_net_fast_path(
    linear,
    inputs: mx.array,
    mask: Optional[mx.array],
    cache: Optional[Any],
    use_kernel: bool | None = None,
) -> mx.array:
    """Pre-target-verify Qwen3.5 GDN path for ordinary decode.

    Upstream mlx-vlm added target-verification and ragged-batch helpers to this
    layer. Those are needed for MTP/ragged server decode, but the decode-only
    conv/projection helpers are slower for mlx-engine's common batch-size-one
    path. This preserves the older plain path when no target-verify state is in
    play.
    """
    B, S, _ = inputs.shape

    mixed_qkv = linear.in_proj_qkv(inputs)

    z = linear.in_proj_z(inputs)
    z = z.reshape(B, S, -1, linear.head_v_dim)

    b = linear.in_proj_b(inputs)
    a = linear.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
        if conv_state.shape[0] != B:
            conv_state = mx.zeros(
                (B, linear.conv_kernel_size - 1, linear.conv_dim),
                dtype=inputs.dtype,
            )
    else:
        conv_state = mx.zeros(
            (B, linear.conv_kernel_size - 1, linear.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        if mask.shape[0] != B:
            mask = None
        else:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    if state is not None and state.shape[0] != B:
        state = None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        linear.A_log,
        linear.dt_bias,
        state,
        mask,
        use_kernel=not getattr(linear, "training", False)
        if use_kernel is None
        else use_kernel,
    )

    if cache is not None:
        cache[1] = state
        if hasattr(cache, "advance"):
            cache.advance(S)

    out = linear.norm(out, z)
    return linear.out_proj(out.reshape(B, S, -1))


def _has_vlm_qwen3_5_ragged_cache_state(cache) -> bool:
    return (
        getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    )


def _patched_vlm_qwen3_5_is_single_row_batch_cache(cache_entry) -> bool:
    left_padding = getattr(cache_entry, "left_padding", None)
    if not (
        isinstance(left_padding, mx.array)
        and left_padding.ndim > 0
        and left_padding.size == 1
    ):
        return False

    cached = getattr(cache_entry, "_mlx_engine_qwen3_5_single_row_batch_cache", None)
    if cached is None or cached[0] is not left_padding:
        cached = (left_padding, bool(int(left_padding.item()) > 0))
        cache_entry._mlx_engine_qwen3_5_single_row_batch_cache = cached
    return cached[1]


class PatchedDecoderLayer(DecoderLayer):
    """
    DecoderLayer that accepts position_ids and uses MRoPE when they are provided.

    For text-only calls (position_ids=None), delegates to the original
    Qwen3NextAttention and GatedDeltaNet — bit-identical to the unpatched model.

    For vision calls (position_ids provided), uses mlx-vlm's MRoPE attention
    and GatedDeltaNet — bit-identical to the native mlx-vlm model.
    """

    def __init__(self, args, layer_idx):
        super().__init__(args, layer_idx)
        if not self.is_linear:
            rope_params = args.rope_parameters
            self._mrope = Qwen3_5RotaryEmbedding(
                int(self.self_attn.head_dim * rope_params["partial_rotary_factor"]),
                max_position_embeddings=args.max_position_embeddings,
                base=rope_params["rope_theta"],
                mrope_section=rope_params["mrope_section"],
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_linear:
            if position_ids is None:
                # Text-only: use original mlx-lm GatedDeltaNet
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
            else:
                # Vision: use mlx-vlm GatedDeltaNet computation path
                r = self._vlm_gated_delta_net(self.input_layernorm(x), mask, cache)
        elif position_ids is None:
            # Text-only: use original Qwen3NextAttention with nn.RoPE
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            # Vision: apply MRoPE using the original attention module's weights
            r = self._mrope_attention(
                self.input_layernorm(x), mask, cache, position_ids
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

    def _mrope_attention(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: mx.array,
    ) -> mx.array:
        """
        MRoPE attention path, reusing the original attention module's weights.

        Mirrors Qwen3_5Attention.__call__ from mlx-vlm but operates on
        self.self_attn's projections and norms directly.
        """
        attn = self.self_attn
        B, L, D = x.shape

        q_proj_output = attn.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = attn.k_proj(x), attn.v_proj(x)

        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cos, sin = self._mrope(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=attn.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return attn.o_proj(output * mx.sigmoid(gate))

    def _vlm_gated_delta_net(
        self,
        inputs: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        """
        mlx-vlm GatedDeltaNet computation path, reusing self.linear_attn's weights.

        Mirrors Qwen3_5GatedDeltaNet.__call__ from mlx-vlm to produce
        bit-identical output with the native mlx-vlm model.
        """
        linear = self.linear_attn
        B, S, _ = inputs.shape

        mixed_qkv = linear.in_proj_qkv(inputs)

        z = linear.in_proj_z(inputs)
        z = z.reshape(B, S, -1, linear.head_v_dim)

        b = linear.in_proj_b(inputs)
        a = linear.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = mx.zeros(
                    (B, linear.conv_kernel_size - 1, linear.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (B, linear.conv_kernel_size - 1, linear.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
        conv_out = nn.silu(linear.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
                [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
                [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != B:
            state = None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            linear.A_log,
            linear.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            # Follow mlx-vlm cache advance logic (conditional cache.advance call)
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/58e2435/mlx_vlm/models/qwen3_5/language.py#L350-L353
            # mlx-lm is not conditional
            # ref: https://github.com/ml-explore/mlx-lm/blob/aa4f880/mlx_lm/models/qwen3_5.py#L196-L198
            if hasattr(cache, "advance"):
                cache.advance(S)

        out = linear.norm(out, z)
        return linear.out_proj(out.reshape(B, S, -1))


class PatchedQwen3_5TextModel(Qwen3_5TextModel):
    """
    Qwen3_5TextModel with MRoPE position state management.

    Adds position_ids and rope_deltas attributes that can be set externally
    (by the vision add-on) before generation. During forward passes, computes
    the appropriate position_ids from this state and threads them to decoder layers.

    Position state logic (ported from mlx_vlm LanguageModel.__call__):
    - Both None: text-only path, computes sequential 3D positions from cache state
    - position_ids set: prefill, slices stored positions by cache offset
    - Only rope_deltas set: autoregressive generation, computes from delta
    """

    def __init__(self, args):
        super().__init__(args)
        self.position_ids = None
        self.rope_deltas = None

    def reset_mrope_state(self):
        """
        Reset MRoPE position state.

        Called by the vision add-on's clear_prediction_state before every
        prediction. For vision requests, compute_embeddings sets fresh
        state immediately after.
        """
        self.position_ids = None
        self.rope_deltas = None

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

        Inspired by mlx_vlm.models.qwen3_5.language.LanguageModel.__call__ .
        """
        cache_offset = 0
        cache_offset_scalar = 0
        if cache is not None and cache[self.fa_idx] is not None:
            offset = cache[self.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
                cache_offset_scalar = offset
            elif isinstance(offset, mx.array) and offset.ndim == 0:
                cache_offset = offset.item()
                cache_offset_scalar = cache_offset
            elif isinstance(offset, mx.array):
                cache_offset = offset
                cache_offset_scalar = offset[0].item()

        batch_size, seq_length = inputs.shape

        # Text-only path; no MRoPE state was injected for this call.
        # Return None so PatchedDecoderLayer uses the original nn.RoPE path.
        if self.position_ids is None and self.rope_deltas is None:
            return None
        if self.position_ids is not None and self.rope_deltas is None:
            raise ValueError(
                "MRoPE state is inconsistent: position_ids are set but rope_deltas "
                "are missing."
            )

        # This branch is taken for vision requests while the current call is still
        # consuming prompt tokens that were part of the original multimodal prompt.
        # That includes chunked prompt prefill where cache_offset > 0 but we are still
        # inside the image span, and it also covers callers whose chunk begins inside
        # the stored prompt positions but extends past them by stitching a stored
        # prefix to a sequential tail.
        if self.position_ids is not None:
            stored_seq_length = self.position_ids.shape[2]
            if cache_offset_scalar < stored_seq_length:
                stored_end = min(cache_offset_scalar + seq_length, stored_seq_length)
                stored_positions = self.position_ids[
                    :, :, cache_offset_scalar:stored_end
                ]
                if stored_end - cache_offset_scalar == seq_length:
                    return stored_positions

                tail_seq_length = seq_length - (stored_end - cache_offset_scalar)
                tail_positions = self._compute_sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=tail_seq_length,
                    start_offset=stored_seq_length,
                    rope_deltas=self.rope_deltas,
                )
                return mx.concatenate([stored_positions, tail_positions], axis=2)

        return self._compute_sequential_position_ids(
            batch_size=batch_size,
            seq_length=seq_length,
            start_offset=cache_offset,
            rope_deltas=self.rope_deltas,
        )

    def _compute_sequential_position_ids(
        self,
        *,
        batch_size: int,
        seq_length: int,
        start_offset: int | mx.array,
        rope_deltas: Optional[mx.array] = None,
    ) -> mx.array:
        """Build sequential 3D positions from cache offsets and optional
        rope_deltas once prompt positions have been exhausted."""
        delta = mx.array(start_offset)
        if rope_deltas is not None:
            delta = delta + rope_deltas

        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if delta.ndim == 0:
            delta = mx.broadcast_to(delta.reshape(1, 1), (batch_size, 1))
        elif delta.ndim == 1:
            delta = delta[:batch_size].reshape(-1, 1)
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, 1))
        else:
            delta = delta[:batch_size]
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, delta.shape[1]))

        position_ids = mx.add(position_ids, delta)[None, ...]
        return mx.broadcast_to(position_ids, (3, batch_size, seq_length))


def _patched_vlm_qwen3_5_attention_init(self, args):
    OriginalVlmQwen3_5AttentionInit(self, args)
    rope_params = args.rope_parameters
    self.rope = initialize_rope(
        int(self.head_dim * rope_params["partial_rotary_factor"]),
        base=rope_params["rope_theta"],
        traditional=False,
        scaling_config=rope_params,
        max_position_embeddings=args.max_position_embeddings,
    )


def _patched_vlm_qwen3_5_attention_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
    position_ids: Optional[mx.array] = None,
    position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    target_verify: bool = False,
) -> mx.array:
    if (
        position_ids is not None
        or position_embeddings is not None
        or target_verify
        or (isinstance(mask, str) and mask == "left_padded_decode")
    ):
        return OriginalVlmQwen3_5AttentionCall(
            self,
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            target_verify=target_verify,
        )

    return Qwen3NextAttention.__call__(self, x, mask, cache)


def _patched_vlm_qwen3_5_gated_delta_net_call(
    self,
    inputs: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
    gdn_sink: Optional[list] = None,
    target_verify: bool = False,
) -> mx.array:
    if (
        target_verify
        or gdn_sink is not None
        or inputs.shape[1] != 1
        or cache is None
        or _has_vlm_qwen3_5_ragged_cache_state(cache)
    ):
        return OriginalVlmQwen3_5GatedDeltaNetCall(
            self,
            inputs,
            mask=mask,
            cache=cache,
            gdn_sink=gdn_sink,
            target_verify=target_verify,
        )

    return _vlm_qwen3_5_gated_delta_net_fast_path(
        self,
        inputs,
        mask,
        cache,
        use_kernel=not self.training,
    )


def _is_vlm_qwen3_5_text_only(
    *,
    mask,
    position_ids,
    pixel_values,
    image_grid_thw,
    video_grid_thw,
    capture_layer_ids,
    rope_deltas_kw,
    stored_position_ids,
    stored_rope_deltas,
) -> bool:
    return (
        mask is None
        and position_ids is None
        and pixel_values is None
        and image_grid_thw is None
        and video_grid_thw is None
        and capture_layer_ids is None
        and rope_deltas_kw is None
        and stored_position_ids is None
        and stored_rope_deltas is None
    )


def _vlm_qwen3_5_batched_left_padding_position_ids(
    cache,
    fa_idx: int,
    batch_size: int,
    seq_length: int,
    dtype,
) -> mx.array | None:
    if cache is None or fa_idx >= len(cache) or seq_length != 1:
        return None
    fa_cache = cache[fa_idx]
    left_padding = getattr(fa_cache, "left_padding", None)
    offset = getattr(fa_cache, "offset", None)
    if (
        not isinstance(left_padding, mx.array)
        or left_padding.ndim == 0
        or int(left_padding.max().item()) <= 0
        or not isinstance(offset, mx.array)
        or offset.ndim == 0
    ):
        return None

    offsets = mx.maximum(offset[:batch_size], 0).reshape(-1, 1)
    position_ids = mx.arange(seq_length, dtype=dtype).reshape(1, -1)
    position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
    return mx.add(position_ids, offsets)


def _patched_vlm_qwen3_5_language_model_call(
    self,
    inputs: mx.array,
    inputs_embeds: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    cache=None,
    **kwargs,
):
    position_ids = kwargs.get("position_ids")
    pixel_values = kwargs.get("pixel_values")
    image_grid_thw = kwargs.get("image_grid_thw")
    video_grid_thw = kwargs.get("video_grid_thw")
    capture_layer_ids = kwargs.get("capture_layer_ids")
    rope_deltas_kw = kwargs.get("rope_deltas")
    text_only = _is_vlm_qwen3_5_text_only(
        mask=mask,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        capture_layer_ids=capture_layer_ids,
        rope_deltas_kw=rope_deltas_kw,
        stored_position_ids=getattr(self, "_position_ids", None),
        stored_rope_deltas=getattr(self, "_rope_deltas", None),
    )

    if text_only:
        position_ids = _vlm_qwen3_5_batched_left_padding_position_ids(
            cache,
            self.model.fa_idx,
            inputs.shape[0],
            inputs.shape[1],
            inputs.dtype,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

    if text_only and position_ids is None:
        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=None,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(
            logits=out,
            hidden_states=None,
            gdn_states=None,
        )

    if rope_deltas_kw is not None:
        # mlx-vlm gates kwarg rope_deltas on this side state during decode.
        self._rope_deltas = rope_deltas_kw

    return OriginalVlmQwen3_5LanguageModelCall(
        self,
        inputs,
        inputs_embeds=inputs_embeds,
        mask=mask,
        cache=cache,
        **kwargs,
    )


def apply_patches():
    """
    Apply Qwen3.5 MRoPE patches.
    """
    import mlx_lm.models.qwen3_5

    mlx_lm.models.qwen3_5.DecoderLayer = PatchedDecoderLayer
    mlx_lm.models.qwen3_5.Qwen3_5TextModel = PatchedQwen3_5TextModel

    VlmQwen3_5Attention.__init__ = _patched_vlm_qwen3_5_attention_init
    VlmQwen3_5Attention.__call__ = _patched_vlm_qwen3_5_attention_call
    VlmQwen3_5GatedDeltaNet.__call__ = _patched_vlm_qwen3_5_gated_delta_net_call
    VlmQwen3_5LanguageModel.__call__ = _patched_vlm_qwen3_5_language_model_call
    vlm_qwen3_5_language._is_single_row_batch_cache = (
        _patched_vlm_qwen3_5_is_single_row_batch_cache
    )
    vlm_qwen3_5_language._qwen3_5_ragged_decode_attention = (
        _patched_vlm_qwen3_5_ragged_decode_attention
    )
