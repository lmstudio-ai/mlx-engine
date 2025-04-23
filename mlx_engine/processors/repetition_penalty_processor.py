from typing import Callable
import mlx.core as mx


class RepetitionPenaltyProcessor:
    def __init__(
        self,
        original_repetition_penalty_processor: Callable[[mx.array, mx.array], mx.array],
        token_history: list[int],
        repetition_context_size: int = None,
    ):
        self.token_history = token_history
        self.original_processor = original_repetition_penalty_processor
        self.repetition_context_size = repetition_context_size

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply repetition penalty processing to the logits.

        Args:
            tokens: The token array of shape [batch_size, seq_len]
            logits: The logits array of shape [batch_size, vocab_size]

        Returns:
            The processed logits array
        """
        # append the required historical tokens to the start of the current tokens so that the
        # repetition penalty accounts for tokens that have already been cached
        num_tokens_to_prepend_from_history = max(
            self.repetition_context_size - len(tokens), 0
        )
        historical_tokens = mx.array(
            self.token_history[-num_tokens_to_prepend_from_history:]
        )
        all_tokens_to_consider = mx.concat([historical_tokens, tokens])
        result = self.original_processor(all_tokens_to_consider, logits)
        assert len(result) <= self.repetition_context_size
        return result
