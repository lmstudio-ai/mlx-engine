from typing import List, Optional, Any

from mlx_lm.models.cache import RotatingKVCache, KVCache
import mlx.core as mx
import mlx.nn as nn
import sys


# unfortunate that this is hardcoded but what else is one to do
MAYBE_ATTN_NAMES = ["self_attn", "attention", "attn", "mixer", "norm_attn_norm"]
MAYBE_ROPE_NAMES = ["rope", "rotary_emb"]


# TODO(christian-lms): stop doing me
def cat(v: mx.array):
    """Alias for mx.concatenate(v, axis=2) since that's used all over the place"""
    return mx.concatenate(v, axis=2)


class ShiftingKVCache(RotatingKVCache):
    def __init__(self, max_size=256, keep=0, step=256):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0
        self.reuse_queue = []

    # TODO(christian-lms): does it matter if you don't change offsets?
    def _trim(self, trim_size, append_k=None, append_v=None) -> None:
        k = self.keys
        v = self.values
        if k is None or v is None:
            return
        assert k.shape == v.shape
        if trim_size > 0:
            k_cat = [
                k[..., : self.keep, :],
                k[..., trim_size + self.keep :, :],
            ]
            v_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
            # TODO(christian-lms): try removing me. if it seems fine then revert
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
        self.keys, self.values = cat(k_cat), cat(v_cat)

    def _temporal_order(self) -> None:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        k = self.keys
        v = self.values
        if k is None or v is None:
            return
        assert k.shape == v.shape
        if self._idx == v.shape[2]:
            pass
        elif self._idx < self.offset:
            shift_by = self.keep - self._idx  # intentionally negative!!!
            assert shift_by <= 0
            # TODO(christian-lms): try removing me. if it seems fine then revert
            self.offset += shift_by
            kcat = cat(
                [
                    k[..., : self.keep, :],
                    k[..., self._idx :, :],
                    k[..., self.keep : self._idx, :],
                ],
            )
            vcat = cat(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
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

            key_segments.append(self.keys[..., reuse_start_idx:reuse_end_idx, :])
            value_segments.append(self.values[..., reuse_start_idx:reuse_end_idx, :])

            current_pos = write_start_idx + reuse_length
            self.offset += shift_by

        self.keys, self.values = cat(key_segments), cat(value_segments)

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

        new_length = max(self.keys.shape[2] - n, 0)
        self.keys = self.keys[..., :new_length, :]
        self.values = self.values[..., :new_length, :]

        # TODO(christian-lms): maybe this is wrong??? maybe you have bigger problems elsewhere
        self.offset = new_length
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
            self._trim(trim_size, append_k=keys, append_v=values)
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
                self.keys = cat([self.keys, new_k])
                self.values = cat([self.values, new_v])
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
        print(f"setting keep to {keep} with offset {self.offset} and idx {self._idx}", file=sys.stderr)
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
            ``make_cache`` method, a ``ShiftingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()
    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [ShiftingKVCache(max_size=max_kv_size, keep=keep) for _ in range(num_layers)]
    else:
        return [KVCache() for _ in range(num_layers)]
