"""Module for processing and handling stop sequences in token generation."""

from typing import List, Literal, NamedTuple, Optional, Sequence


StopReason = Literal["eos_token", "stop_string"]

StopProcessorStatus = Literal["full_stop", "partial_match", "no_match"]


class StopProcessorResult(NamedTuple):
    """Result of stop sequence processing containing status and details."""

    status: StopProcessorStatus
    stop_reason: Optional[StopReason]
    stop_tokens: Optional[List[int]]


class StopProcessor:
    """Statefully processes tokens to check for stop sequences during generation."""

    def __init__(self, tokenizer, stop_sequences: List[List[int]]):
        """
        Args:
            tokenizer: Tokenizer instance for encoding/decoding text
            stop_sequences: List of token sequences that signal generation should stop
        """
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.tokens_to_check_for_stopping = []
        self.stop_sequence_suffix = None

    def process_token(self, tokens: int) -> StopProcessorResult:
        """Process a new token and check if generation should stop.

        Args:
            tokens: New token to process

        Returns:
            StopProcessorResult indicating if/how generation should stop
        """
        self.tokens_to_check_for_stopping.append(tokens)
        stopping_criteria_result = stopping_criteria(
            self.tokens_to_check_for_stopping,
            self.stop_sequences,
            self.tokenizer,
        )
        status = stopping_criteria_result.status
        if status == "full_stop":
            # on a full stop, save the stop sequence suffix to allow client to handle it
            # how they'd like (i.e., trim, not trim, etc...)
            if stopping_criteria_result.stop_tokens:
                trim_length = len(stopping_criteria_result.stop_tokens)
                self.stop_sequence_suffix = self.tokenizer.decode(
                    self.tokens_to_check_for_stopping[-trim_length:]
                )
            return StopProcessorResult(
                "full_stop",
                stopping_criteria_result.stop_reason,
                stopping_criteria_result.stop_tokens,
            )
        elif status == "partial_match":
            return StopProcessorResult("partial_match", None, None)
        else:
            # if no partial match, can clear the buffer of tokens we need to check for stopping
            self.tokens_to_check_for_stopping = []
            return StopProcessorResult("no_match", None, None)


# Adapted from mlx_lm.utils.server.py
# https://github.com/ml-explore/mlx-examples/blob/605c4854f1547e8eb0ef3f9c9d81c8aef3196c15/llms/mlx_lm/server.py#L43-L74
def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    tokenizer,
) -> StopProcessorResult:
    """Check if token generation should stop.

    Checks three conditions in order:
    1. End of sequence token from tokenizer
    2. Exact token match with stop sequences
    3. Full match between decoded text and any decoded stop sequence
    4. Partial match with any decoded stop sequence
    """
    # Check for end of sequence token
    if tokens and tokens[-1] == tokenizer.eos_token_id:
        return StopProcessorResult(
            status="full_stop",
            stop_reason="eos_token",
            stop_tokens=[tokenizer.eos_token_id],
        )

    # Check for exact token sequence matches
    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids) and tokens[-len(stop_ids) :] == stop_ids:
            return StopProcessorResult(
                status="full_stop",
                stop_reason="stop_string",
                stop_tokens=stop_ids,
                is_partial_match=False,
            )

    # Check for full matches in decoded text
    tokens_str = tokenizer.decode(tokens)
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        if stop_str in tokens_str:
            return StopProcessorResult(
                status="full_stop", stop_reason="stop_string", stop_tokens=stop_ids
            )

    # Check for partial matches only if no full matches were found
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        if sequence_overlap(tokens_str, stop_str):
            return StopProcessorResult(
                status="partial_match", stop_reason=None, stop_tokens=None
            )

    # No matches found
    return StopProcessorResult(status="no_match", stop_reason=None, stop_tokens=None)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))
