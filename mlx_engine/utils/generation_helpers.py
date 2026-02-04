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
    if stop_strings is not None and len(stop_strings) > 0:
        return StopStringProcessor(stop_strings, tokenizer)
    return None


def process_stop_string_check(
    stop_string_processor: Optional[StopStringProcessor], token: int
) -> Tuple[bool, bool, Optional[StopStringProcessorResult]]:
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
