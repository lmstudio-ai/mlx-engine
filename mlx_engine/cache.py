from typing import List, Optional, Any

from mlx_lm.models.cache import RotatingKVCache, KVCache
import mlx.nn as nn


class AlwaysTrimmableKVCache(RotatingKVCache):
    """A KV cache that can always be trimmed.

    The MLX-LM implementation of the RotatingKVCache does not allow trimming
    the cache once the maximum KV size has been exceeded, which results in
    the cache being nuked every time this happens. This forces the entire context
    to be reprocessed regularly, which is not ideal for performance. This KV cache
    allows trimming the cache at any time, which circumvents this issue.
    See https://github.com/lmstudio-ai/mlx-engine/issues/177 for more details.
    """

    def trim(self, n) -> int:
        # trim must not respect keep: we always receive some value for keep, but
        # when initially processing the prompt, it may be that the common prefix
        # is shorter than keep. in that case we must trim to the common prefix length,
        # which violates keep. keep is only used for the cache rotation when exceeding
        # the context length mid-generation to ensure we don't lose the common prefix.
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

    See https://github.com/ml-explore/mlx-lm/blob/fd9b1909636d634ac2b848248b05939c9fbfbe19/mlx_lm/models/cache.py#L10
    for the MLX-LM implementation. This is a temporary extension to support more flexible
    trimming than MLX-LM's original RotatingKVCache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``AlwaysTrimmableKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()
    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            AlwaysTrimmableKVCache(max_size=max_kv_size, keep=keep)
            for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]
