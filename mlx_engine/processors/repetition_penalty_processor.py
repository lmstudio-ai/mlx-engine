import mlx.core as mx
from mlx_lm.sample_utils import make_repetition_penalty

"""
Wrapper for the standard mlx-lm repetition penalty processor
ref: https://github.com/ml-explore/mlx-lm/blob/69195f8632869d35306d085de7dc4e7d6954baac/mlx_lm/sample_utils.py#L245-L255

This wrapper enables the repetition penalty processor to take into account the tokens that have already been cached,
without the need for recomputing the logits for those tokens.
"""


class RepetitionPenaltyProcessor:
    def __init__(
        self,
        token_history: list[int],
        repetition_penalty: float,
        repetition_context_size: int,
    ):
        self.token_history = token_history
        self.repetition_context_size = repetition_context_size
        self.repetition_penalty_function = make_repetition_penalty(
            repetition_penalty, repetition_context_size
        )

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply repetition penalty to the logits, accounting for tokens that have already been processed within
        the same prediction.

        Args:
            tokens: The tokens to be processed.
            logits: The logits to be processed.
        """
        # append historical tokens s.t. repetition penalty accounts tokens that have already been processed in this gen
        num_tokens_to_prepend_from_history = max(
            self.repetition_context_size - len(tokens), 0
        )
        historical_tokens = (
            self.token_history[-num_tokens_to_prepend_from_history:]
            if num_tokens_to_prepend_from_history > 0
            else []
        )
        historical_tokens_mx = mx.array(
            historical_tokens,
            dtype=mx.int64,
        )
        all_tokens_to_consider = mx.concat([historical_tokens_mx, tokens])
        result = self.repetition_penalty_function(all_tokens_to_consider, logits)
        return result
