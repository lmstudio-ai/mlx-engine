"""
Helper functions for text generation that are shared between sequential and batched generation.

These functions handle common setup and processing tasks like sampler creation,
logits processor configuration, and stop condition detection.
"""

from typing import Optional, List, Tuple
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler
from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)
from mlx_engine.stop_string_processor import (
    StopStringProcessor,
    StopStringProcessorResult,
)
from mlx_engine.utils.generation_result import GenerationStopCondition
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor

MAX_TOP_LOGPROBS = 10


def setup_repetition_penalty(
    repetition_penalty: Optional[float], repetition_context_size: Optional[int]
) -> dict:
    """
    Setup repetition penalty parameters.

    Args:
        repetition_penalty: Penalty factor for repeated tokens
        repetition_context_size: Number of previous tokens to consider

    Returns:
        dict: Dictionary of repetition penalty kwargs
    """
    repetition_penalty_kwargs = {}
    if repetition_penalty is not None:
        repetition_penalty_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            repetition_penalty_kwargs["repetition_context_size"] = (
                repetition_context_size
            )
    return repetition_penalty_kwargs


def setup_logits_processors(
    repetition_penalty: Optional[float],
    repetition_penalty_kwargs: dict,
    prompt_tokens: List[int],
    input_tokens: List[int],
    json_schema: Optional[str],
    tokenizer: TokenizerWrapper,
) -> List:
    """
    Setup logits processors for repetition penalty and JSON schema.

    Args:
        repetition_penalty: Penalty factor for repeated tokens
        repetition_penalty_kwargs: Dictionary of repetition penalty parameters
        prompt_tokens: Full prompt token list
        input_tokens: Input tokens to process
        json_schema: Optional JSON schema for structured output
        tokenizer: The tokenizer instance

    Returns:
        List: List of configured logits processors
    """
    logits_processors = []

    if repetition_penalty and repetition_penalty != 0.0:
        cached_tokens = (
            prompt_tokens[: -len(input_tokens)]
            if len(input_tokens) > 0
            else prompt_tokens
        )
        logits_processors.append(
            RepetitionPenaltyProcessor(
                token_history=cached_tokens, **repetition_penalty_kwargs
            )
        )

    if json_schema is not None:
        logits_processors.append(
            JSONLogitsProcessor(
                json_schema,
                OutlinesTransformerTokenizer(tokenizer._tokenizer),
                tensor_library_name="mlx",
            )
        )

    return logits_processors


def create_sampler(
    temp: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    min_tokens_to_keep: Optional[int],
    top_k: Optional[int],
):
    """
    Create sampler with filtering non-None parameters.

    Args:
        temp: Temperature for sampling
        top_p: Top-p (nucleus) sampling parameter
        min_p: Minimum probability threshold
        min_tokens_to_keep: Minimum number of tokens to keep
        top_k: Top-k sampling parameter

    Returns:
        Sampler function configured with the provided parameters
    """
    return make_sampler(
        **{
            k: v
            for k, v in {
                "temp": temp,
                "top_p": top_p,
                "min_p": min_p,
                "min_tokens_to_keep": min_tokens_to_keep,
                "top_k": top_k,
            }.items()
            if v is not None
        }
    )


def validate_top_logprobs(top_logprobs: Optional[int]) -> int:
    """
    Validate and normalize top_logprobs parameter.

    Args:
        top_logprobs: Number of top token probabilities to return

    Returns:
        int: Normalized top_logprobs value (0 if None)

    Raises:
        ValueError: If top_logprobs exceeds MAX_TOP_LOGPROBS
    """
    if top_logprobs is None:
        top_logprobs = 0
    if top_logprobs > MAX_TOP_LOGPROBS:
        raise ValueError(
            f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
        )
    return top_logprobs


def create_stop_string_processor(
    stop_strings: Optional[List[str]], tokenizer: TokenizerWrapper
) -> Optional[StopStringProcessor]:
    """
    Create stop string processor if stop_strings are provided.

    Args:
        stop_strings: List of strings that trigger generation to stop
        tokenizer: The tokenizer instance

    Returns:
        Optional[StopStringProcessor]: Processor instance or None if no stop strings
    """
    if stop_strings is not None and len(stop_strings) > 0:
        return StopStringProcessor(stop_strings, tokenizer)
    return None


def process_stop_string_check(
    stop_string_processor: Optional[StopStringProcessor], token: int
) -> Tuple[bool, bool, Optional[StopStringProcessorResult]]:
    """
    Process token with stop string processor.

    Args:
        stop_string_processor: The stop string processor instance or None
        token: Token ID to process

    Returns:
        tuple: (should_stop, should_buffer, processor_result)
            - should_stop: True if generation should stop
            - should_buffer: True if we should buffer without yielding
            - processor_result: The processor result if should_stop is True
    """
    if stop_string_processor is None:
        return False, False, None

    result = stop_string_processor.process_token(token)

    if result.status == "full_stop":
        return True, False, result

    if result.status in ("partial_match", "multi_byte"):
        return False, True, None

    return False, False, None


def should_yield_token(
    text: str, token: int, tokenizer: TokenizerWrapper
) -> Tuple[bool, Optional[GenerationStopCondition]]:
    """
    Determine if token should be yielded and create stop condition if EOS.

    Args:
        text: Current generated text segment
        token: Token ID to check
        tokenizer: The tokenizer instance

    Returns:
        tuple: (should_yield, stop_condition)
            - should_yield: True if the token should be yielded
            - stop_condition: GenerationStopCondition if EOS token, None otherwise
    """
    if text or token in tokenizer.eos_token_ids:
        stop_condition = None
        if token in tokenizer.eos_token_ids:
            stop_condition = GenerationStopCondition(
                stop_reason="eos_token",
                stop_string=tokenizer.decode(token),
                stop_tokens=[token],
            )
        return True, stop_condition
    return False, None
