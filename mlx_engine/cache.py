from typing import List, Optional, Any

from mlx_lm.models.cache import RotatingKVCache, KVCache
import mlx.core as mx
import mlx.nn as nn


class ShiftingKVCache(RotatingKVCache):
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