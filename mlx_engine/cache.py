from typing import List, Optional, Any

from mlx_engine.logging import log_warn
from mlx_lm.models.cache import RotatingKVCache, KVCache
import mlx.core as mx
import mlx.nn as nn


# TODO(christian-lms) DO NOT HARDCODE ME (or at least move it somewhere else)
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


# TODO(christian-lms): you end up basically overriding EVERYTHING so maybe decouple
class ShiftingKVCache(RotatingKVCache):
    def __init__(self, rope: nn.Module, max_size=None, keep=0, step=256):
        self._rope = rope
        self.reuse_offset = 0
        self.reuse_queue = []
        super().__init__(max_size, keep, step)

    def rope(self, v: mx.array, shift_by: int) -> mx.array:
        # TODO(christian-lms): this is reeeeeeallllyyyy stupid. spin a proper block impl
        return mx.concatenate(
            [self._rope(v[:, :, i : i + 1, :], shift_by) for i in range(v.shape[2])],
            axis=2,
        )
        
    def rope_if(self, v: mx.array, shift_by: int, do: bool = False) -> mx.array:
        return self.rope(v, shift_by) if do else v

    def is_trimmable(self) -> bool:
        return True

    def _trim(self, trim_size, v, append=None, is_key=False):
        to_cat = []
        shift_by = -trim_size
        if trim_size > 0:
            to_cat = [
                v[..., : self.keep, :],
                self.rope_if(v[..., trim_size + self.keep :, :], shift_by, do=is_key),
            ]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v, is_key=False) -> mx.array:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            shift_by = self.keep - self._idx
            assert shift_by <= 0
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    # N.B. this implicitly assumes the generation has not gone over twice
                    # the size of the rotating section of the cache, in which case the
                    # rotating section would be off by a multiple of (max_kv_size - keep)
                    # depending on how many times it rolled over. I feel like it's pretty
                    # safe to assume that this is a rare case
                    self.rope_if(v[..., self._idx :, :], shift_by, do=is_key),
                    self.rope_if(v[..., self.keep : self._idx, :], shift_by, do=is_key),
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]
    
    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys, is_key=True)
            self.values = self._temporal_order(self.values, is_key=False)

            # The largest size is self.max_size + S to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size
            self.keys = self._trim(trim_size, self.keys, keys, is_key=True)
            self.values = self._trim(trim_size, self.values, values, is_key=False)
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
            self.keys = self._trim(trim_size, self.keys, is_key=True)
            self.values = self._trim(trim_size, self.values, is_key=False)
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

    def reuse_section(
        self, write_start_idx: int, reuse_start_idx: int, reuse_length: int
    ) -> None:
        # offset indices to account for the fact that we move cache elements around
        write_start_idx -= self.reuse_offset
        reuse_start_idx -= self.reuse_offset

        # update position offsets for future reuse sections
        shift_by = write_start_idx - reuse_start_idx
        self.reuse_offset += shift_by

        # queue for reuse: everything is done in one pass at the end in do_reuse
        self.reuse_queue.append((write_start_idx, reuse_start_idx, reuse_length))

    def do_reuse(self) -> None:
        last_i: int = len(self.reuse_queue) - 1

        for i, (write_start_idx, reuse_start_idx, reuse_length) in enumerate(
            self.reuse_queue
        ):
            shift_by: int = write_start_idx - reuse_start_idx
            assert shift_by <= 0
            reuse_end_idx: int = reuse_start_idx + reuse_length

            keys_to_shift = self.keys[..., reuse_start_idx:reuse_end_idx, :]
            values_to_shift = self.values[..., reuse_start_idx:reuse_end_idx, :]

            # perform rope shift
            # N.B. we can also go back to the MLX-native "don't rope shift" method
            # by removing RoPE here and removing the overrides for trim, temporal order
            shifted_keys = self.rope(keys_to_shift, shift_by)

            # restructure cache with mx.concat
            # TODO(christian-lms): maybe it would be better to use inplace ops.
            # look into the mlx docs if that's even a thing
            keycat = [self.keys[..., :write_start_idx, :], shifted_keys]
            valcat = [self.values[..., :write_start_idx, :], values_to_shift]

            # TODO(christian-lms): surely there is a better way to do this?
            # by not re-appending the end at the last one, we truncate the leftovers
            if i != last_i:
                keycat.append(self.keys[..., reuse_end_idx:, :])
                valcat.append(self.values[..., reuse_end_idx:, :])

            self.keys = mx.concatenate(keycat, axis=2)
            self.values = mx.concatenate(valcat, axis=2)

            self.offset -= shift_by
        self.reuse_offset = 0
        self.reuse_queue = []
        # TODO(christian-lms): dunno if this number is correct/reasonable/whatever
        self._idx = self.keys.shape[2]

    def trim(self, n) -> int:
        # TODO(christian-lms): should trim respect keep? currently, no
        n = min(self.offset, n)
        if n <= 0:
            return 0

        # TODO(christian-lms): so you used to need to wrap around because the code
        # didn't know how much it was trying to trim, so it would go over the maximum allowed.
        # but i think this was in large part due to improperly tracking the tokens that were
        # actually in the cache, so this should not be an issue anymore. therefore this trim code
        # will trim exactly n off the end wthout any wrapping around. but you can uncomment the line
        # if it turns out that this assumption is faulty
        if self.offset >= self.max_size:
            self.keys = self._temporal_order(self.keys, is_key=True)
            self.values = self._temporal_order(self.values, is_key=False)
            # n = n % (self.max_size - self.keep)

        # do trim: put us back into the state before the circular buffer is full
        new_length = self.keys.shape[2] - n
        self.keys = self.keys[..., :new_length, :]
        self.values = self.values[..., :new_length, :]

        self.offset -= n
        # TODO(christian-lms): verify that this is reasonable
        self._idx = new_length
        return n


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
