from typing import Callable
import mlx.core as mx

"""
Wrapper for the standard mlx-lm repetition penalty processor
ref: https://github.com/ml-explore/mlx-lm/blob/69195f8632869d35306d085de7dc4e7d6954baac/mlx_lm/sample_utils.py#L245-L255

This wrapper enables the repetition penalty processor to take into account the tokens that have already been cached,
without the need for recomputing the logits for those tokens.
"""


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
        historical_tokens = mx.array(
            self.token_history[-num_tokens_to_prepend_from_history:]
        )
        all_tokens_to_consider = mx.concat([historical_tokens, tokens])
        result = self.original_processor(all_tokens_to_consider, logits)
        return result


def replace_default_repetition_penalty_processor(
    logits_processors: list[Callable],
    prompt_tokens: list[int],
    num_uncached_tokens: int,
    repetition_context_size: int,
) -> bool:
    """
    Replace the default repetition penalty processor in logits_processors
    with a custom one that takes into account the tokens that have already been cached.

    Raises a ValueError if the repetition penalty processor is not found in the logits processors
    when expected.

    Returns True if the repetition penalty processor was found and replaced, False otherwise.
    """
    # exclude the last stream_generate_input tokens from the prompt to get already computed
    already_computed_tokens = prompt_tokens[:-num_uncached_tokens]
    # Find and wrap the repetition penalty processor
    replaced_default_repetition_penalty_processor = False
    for i, processor in enumerate(logits_processors):
        processor_name = (
            processor.__name__ if hasattr(processor, "__name__") else str(processor)
        )
        if processor_name == "repetition_penalty_processor":
            logits_processors[i] = RepetitionPenaltyProcessor(
                original_repetition_penalty_processor=processor,
                token_history=already_computed_tokens,
                repetition_context_size=repetition_context_size,
            )
            replaced_default_repetition_penalty_processor = True
            break
    return replaced_default_repetition_penalty_processor
