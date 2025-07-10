from typing import List, Optional, Any

from mlx_lm.models.cache import RotatingKVCache, KVCache
import mlx.core as mx
import mlx.nn as nn


class ShiftingKVCache(RotatingKVCache):
    def __init__(self, max_size=256, keep=0, step=256):
        self.reuse_queue = []
        super().__init__(max_size=max_size, keep=keep, step=step)

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

            reuse_end_idx = reuse_start_idx + reuse_length
            current_pos = write_start_idx + reuse_length

            key_segments.append(self.keys[..., reuse_start_idx:reuse_end_idx, :])
            value_segments.append(self.values[..., reuse_start_idx:reuse_end_idx, :])

        self.keys = mx.concatenate(key_segments, axis=2)
        self.values = mx.concatenate(value_segments, axis=2)

        # clean up
        self.reuse_queue = []
        self._idx = self.keys.shape[2]
        self.offset = self.keys.shape[2]

    def trim(self, n) -> int:
        # trim must not respect keep
        n = min(self.offset, n)
        if n <= 0:
            return 0

        # put us back into the state before the circular buffer is full
        self.keys = self._temporal_order(self.keys)
        self.values = self._temporal_order(self.values)

        new_length = max(self.keys.shape[2] - n, 0)
        self.keys = self.keys[..., :new_length, :]
        self.values = self.values[..., :new_length, :]

        self.offset = new_length
        self._idx = new_length
        return n

    def set_keep(self, keep):
        # kv must be in temporal order, else we will keep the wrong thing
        if self.keys is not None:
            self.keys = self._temporal_order(self.keys)
        if self.values is not None:
            self.values = self._temporal_order(self.values)
        self.keep = keep

    def is_trimmable(self) -> bool:
        return True


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
        return [
            ShiftingKVCache(max_size=max_kv_size, keep=keep) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]
