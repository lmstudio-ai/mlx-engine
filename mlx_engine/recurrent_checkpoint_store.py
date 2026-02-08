import copy
import logging
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


class RecurrentCheckpointStore:
    """
    LRU-bounded store of (token_prefix -> cache_snapshot) pairs.

    Used for checkpoint-based prefix reuse with hybrid/recurrent models whose
    cache layers are not all trimmable (e.g. Qwen3-Next with ArraysCache).
    Instead of clearing the entire cache on diverging suffix, we save snapshots
    at regular intervals during prefill and restore the longest matching one.
    """

    def __init__(self, max_checkpoints: int = 8):
        self._max_checkpoints = max_checkpoints
        # OrderedDict of tuple(tokens) -> List[cache_layer_state]
        # Most-recently-used entries are at the end.
        self._store: OrderedDict[tuple, List[Any]] = OrderedDict()

    def save(self, tokens: mx.array, cache: List[Any]) -> None:
        """
        Save a deep copy of the cache state keyed by the token prefix.

        If the key already exists, this updates LRU order only (no re-copy).

        Args:
            tokens: The token prefix processed so far.
            cache: The model cache state to snapshot.
        """
        key = tuple(tokens.tolist())

        if key in self._store:
            # Already stored — just move to most-recently-used
            self._store.move_to_end(key)
            return

        # Deep copy to get an independent snapshot
        snapshot = copy.deepcopy(cache)
        mx.eval([c.state for c in snapshot])

        self._store[key] = snapshot

        # Evict oldest if over capacity
        while len(self._store) > self._max_checkpoints:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(
                f"Evicted checkpoint at position {len(evicted_key)} (LRU)"
            )

    def find_longest_prefix(
        self, tokens: mx.array
    ) -> Optional[Tuple[int, List[Any]]]:
        """
        Find the checkpoint with the longest token prefix matching the start of `tokens`.

        Returns a deep copy of the cached state so the caller gets an independent copy.

        Args:
            tokens: The full token sequence to match against.

        Returns:
            A tuple of (prefix_length, cache_deep_copy) for the longest match,
            or None if no checkpoint shares a prefix with `tokens`.
        """
        tokens_tuple = tuple(tokens.tolist())
        best_key: Optional[tuple] = None
        best_len = 0

        for key in self._store:
            key_len = len(key)
            if key_len > len(tokens_tuple):
                continue
            if key_len <= best_len:
                continue
            # Check if key is a prefix of tokens
            if tokens_tuple[:key_len] == key:
                best_key = key
                best_len = key_len

        if best_key is None:
            return None

        # Move to most-recently-used
        self._store.move_to_end(best_key)

        # Deep copy so caller gets independent state
        restored = copy.deepcopy(self._store[best_key])
        mx.eval([c.state for c in restored])

        logger.info(f"Restored checkpoint at position {best_len}")
        return best_len, restored

    def clear(self) -> None:
        """Remove all checkpoints."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
