"""Token penalty processor with KV cache awareness."""

from collections.abc import Callable

import mlx.core as mx


class TokenPenaltyProcessor:
    # Prepends cached prefix tokens so the penalty window spans the full context,
    # not just the tokens generated in the current turn.

    def __init__(
        self,
        penalty_fn: Callable[[mx.array, mx.array], mx.array],
        token_history: list[int],
        context_size: int,
    ):
        self.token_history = token_history
        self.context_size = context_size
        self._penalty_fn = penalty_fn

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        num_to_prepend = max(self.context_size - len(tokens), 0)
        historical = (
            self.token_history[-num_to_prepend:] if num_to_prepend > 0 else []
        )
        all_tokens = mx.concat([mx.array(historical, dtype=mx.int64), tokens])
        return self._penalty_fn(all_tokens, logits)
