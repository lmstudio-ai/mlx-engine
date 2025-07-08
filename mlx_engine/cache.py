from typing import List, Optional, Any

from mlx_engine.logging import log_warn
from mlx_lm.models.cache import _BaseCache, KVCache
import mlx.core as mx
import mlx.nn as nn


# unfortunate that this is hardcoded but what else is one to do
MAYBE_ATTN_NAMES = ["self_attn", "attention", "attn", "mixer", "norm_attn_norm"]
MAYBE_ROPE_NAMES = ["rope", "rotary_emb"]


def _maybe_get_rope(layer: nn.Module) -> Optional[nn.Module]:
    for maybe_rope_name in MAYBE_ROPE_NAMES:
        if hasattr(layer, maybe_rope_name):
            # found it
            return getattr(layer, maybe_rope_name)
    for maybe_attn_name in MAYBE_ATTN_NAMES:
        if hasattr(layer, maybe_attn_name):
            # move down one level
            return _maybe_get_rope(getattr(layer, maybe_attn_name))
    # no dice
    return None


def maybe_get_rope(model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
    """Attempt to find the RoPE module from a layer of an MLX-LM LLM.

    Args:
        model (nn.Module): The LLM to search for the RoPE modules of.
        layer_idx (int): The layer of the LLM to get the RoPE module from.

    Returns:
        Optional[nn.Module]: The RoPE module if found, else None
    """
    # we can assume model has attribute layers because make_prompt_cache does
    if layer_idx > len(model.layers):
        return None
    layer = model.layers[layer_idx]
    if not isinstance(layer, nn.Module):
        return None
    return _maybe_get_rope(layer)


class ShiftingKVCache(_BaseCache):
    def __init__(self, rope: nn.Module, max_size=256, keep=0, step=256):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0
        self._rope = rope
        self.reuse_queue = []

    def rope(self, v: mx.array, shift_by: int) -> mx.array:
        # you'd think this is inefficient, but it seems faster than spinning
        # a custom implementation somehow. also it allows us to easily use the
        # sustk scaled rope/yarn/llama3 rope impls in mlx_lm without having to
        # spin a custom implementation for those too (and any future rope variants)
        return mx.concatenate(
            [self._rope(v[:, :, i : i + 1, :], shift_by) for i in range(v.shape[2])],
            axis=2,
        )

    def _trim(
        self, trim_size, append_k=None, append_v=None
    ) -> None:
        k = self.keys
        v = self.values
        assert k.shape == v.shape
        shift_by = -trim_size
        if trim_size > 0:
            k_cat = [
                k[..., : self.keep, :],
                self.rope(k[..., trim_size + self.keep :, :], shift_by),
            ]
            v_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
            self.offset -= trim_size
        else:
            k_cat = [k]
            v_cat = [v]
        if append_k is not None:
            assert append_v is not None
            k_cat.append(append_k)
            v_cat.append(append_v)
        if append_v is not None:
            assert append_k is not None
            # already done
        self.keys, self.values = mx.concatenate(k_cat, axis=2), mx.concatenate(v_cat, axis=2)

    def _temporal_order(self) -> None:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        k = self.keys
        v = self.values
        assert k.shape == v.shape
        if self._idx == v.shape[2]:
            pass
        elif self._idx < self.offset:
            shift_by = self.keep - self._idx  # intentionally negative!!!
            assert shift_by <= 0
            self.offset += shift_by
            kcat = mx.concatenate(
                [
                    k[..., : self.keep, :],
                    # N.B. this implicitly assumes the generation has not gone over twice
                    # the size of the rotating section of the cache, in which case the
                    # rotating section would be off by a multiple of (max_kv_size - keep)
                    # depending on how many times it rolled over. I feel like it's pretty
                    # safe to assume that this is a rare case
                    self.rope(k[..., self._idx :, :], shift_by),
                    self.rope(k[..., self.keep : self._idx, :], shift_by),
                ],
                axis=2,
            )
            vcat = mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
            self.keys, self.values = kcat, vcat
        else:
            self.keys, self.values = k[..., : self._idx, :], v[..., : self._idx, :]

    def reuse_section(
        self, write_start_idx: int, reuse_start_idx: int, reuse_length: int
    ) -> None:
        # queue for reuse: everything is done in one pass at the end in do_reuse
        self.reuse_queue.append((write_start_idx, reuse_start_idx, reuse_length))

    def do_reuse(self) -> None:
        if not self.reuse_queue:
            return

        # just in case, sort in write order
        self.reuse_queue.sort(key=lambda x: x[0])

        key_segments = []
        value_segments = []
        current_pos = 0

        for write_start_idx, reuse_start_idx, reuse_length in self.reuse_queue:
            # add any gap before this write position
            if current_pos < write_start_idx:
                key_segments.append(self.keys[..., current_pos:write_start_idx, :])
                value_segments.append(self.values[..., current_pos:write_start_idx, :])

            # add the reused segment with RoPE shift
            shift_by = write_start_idx - reuse_start_idx  # intentionally negative!!!
            reuse_end_idx = reuse_start_idx + reuse_length

            keys_to_reuse = self.keys[..., reuse_start_idx:reuse_end_idx, :]
            values_to_reuse = self.values[..., reuse_start_idx:reuse_end_idx, :]

            # only keys require rope
            shifted_keys = self.rope(keys_to_reuse, shift_by)

            key_segments.append(shifted_keys)
            value_segments.append(values_to_reuse)

            current_pos = write_start_idx + reuse_length
            self.offset += shift_by

        self.keys = mx.concatenate(key_segments, axis=2)
        self.values = mx.concatenate(value_segments, axis=2)

        # clean up
        self.reuse_queue = []
        self._idx = self.keys.shape[2]
        self.offset = self.keys.shape[2]

    def trim(self, n) -> int:
        # trim does not respect keep and it will stay this way
        n = min(self.offset, n)
        if n <= 0:
            return 0

        # do trim: put us back into the state before the circular buffer is full
        self._temporal_order()
        new_length = self.keys.shape[2] - n
        self.keys = self.keys[..., :new_length, :]
        self.values = self.values[..., :new_length, :]

        self.offset -= n
        self._idx = new_length
        return n

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to preserve context
            self._temporal_order()

            # The largest size is self.max_size + S to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size
            self._trim(
                trim_size, append_k=keys, append_v=values
            )
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self._trim(
                trim_size,
            )
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def set_keep(self, keep):
        # kv must be in temporal order, else we will keep the wrong thing
        self._temporal_order()
        self.keep = keep

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.keep, self.max_size, self.step, self.offset, self._idx))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.step, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self) -> bool:
        return True

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> Any:
        raise NotImplementedError("ShiftingKVCache Quantization NYI")


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
    keep: int = 4,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.
    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.
    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``TrimmableRotatingKVCache`` is used
            with a maximum size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        # TODO(christian-lms): gah what are you gonna do about models that do this
        # afm7 baichuan_m1 cohere2 gemma3(+friends) llama4 mamba plamo2 recurrent_gemma
        # m1 mamba plamo2 recurrent_gemma are hybrid
        # - afm7 is trivially overridable
        # - cohere2 is swa on some layers but can probably be overridden
        # - gemma3 see cohere2
        # - llama4 uses chunked kv on some layers but can maybe be overridden
        #   though these layers don't have rope modules

        # try to get the model name from model.args.model_type but i suppose this will
        # not always work. that or literally model.__name__ hopefully
        return model.make_cache()
    num_layers = len(model.layers)
    if max_kv_size is not None:
        cache = []
        for layer in range(num_layers):
            rope = maybe_get_rope(model, layer)
            # TODO(christian-lms): it is known that this will fail for some models
            # like llama4 which has no rope module for every fourth layer.
            # this will be figured out Later(tm) once the initial functionality works
            if rope is None:
                log_warn(
                    "Attempted to build a KV cache of shiftable caches, but found"
                    f"None at layer {layer} of model {model}"
                )
                return [KVCache() for _ in range(num_layers)]
            cache.append(ShiftingKVCache(rope, max_size=max_kv_size, keep=keep))
        return cache
    else:
        return [KVCache() for _ in range(num_layers)]
