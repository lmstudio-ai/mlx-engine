"""Module for processing and handling stop sequences in token generation."""

from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple
import sys


# used internally and in generate.py
StopProcessorStatus = Literal["full_stop", "partial_match", "no_match"]

StopReason = Literal["eos_token", "stop_string"]

class GenerationStopCondition(NamedTuple):
    stop_reason: StopReason
    stop_string: str
    stop_tokens: List[int]

# only used internally
class StopStringProcessorResult(NamedTuple):
    """Result of stop string processing containing status and details."""
    status: StopProcessorStatus
    # the ids that make up the stop string that was found
    stop_tokens: Optional[List[int]]


class StopStringProcessor:
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
        self.stop_sequence_suffix: str = None

    def process_token(self, tokens: int) -> StopStringProcessorResult:
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
            # on a full stop, save the string stop sequence suffix to allow client to handle it
            # how they'd like (i.e., trim, not trim, etc...)
            stop_tokens_found = stopping_criteria_result.stop_tokens
            if stop_tokens_found:
                stop_string = self.tokenizer.decode(
                    stop_tokens_found
                )

                buffered_string = self.tokenizer.decode(
                    self.tokens_to_check_for_stopping
                )

                start_of_stop_string_idx = buffered_string.find(stop_string)

                if start_of_stop_string_idx == -1:
                    sys.stderr.write(
                        f"[StopStringProcessor] Could not find stop string in buffered tokens, "
                        "even though a full stop was detected. Not setting stop_sequence_suffix."
                    )
                else:
                    trim_length = len(buffered_string) - start_of_stop_string_idx
                    self.stop_sequence_suffix = buffered_string[-trim_length:]
            return StopStringProcessorResult(status="full_stop", stop_tokens=stop_tokens_found)
        elif status == "partial_match":
            return StopStringProcessorResult(status="partial_match", stop_tokens=None)
        else:
            # if no partial match, can clear the buffer of tokens we need to check for stopping
            self.tokens_to_check_for_stopping = []
            return StopStringProcessorResult(status="no_match", stop_tokens=None)

    # TODO(matt): this should cease to exist. the client should do this, since this processor
    #  must only be in the scope of custom stop strings and not eos tokens/other that the stream_generate
    #  function already handles
    def finalize(
        self, last_segment: str, stop_processor_result: StopStringProcessorResult
    ) -> Tuple[str, Optional[GenerationStopCondition]]:
        print(f"self.stop_sequence_suffix: {self.stop_sequence_suffix}")
        if last_segment:
            if self.stop_sequence_suffix is not None:
                last_segment = last_segment[: -len(self.stop_sequence_suffix)]

        # build up the final generation stop condition safely, although stop_processor_result
        # should always be set at this point
        generation_stop_condition = None
        if stop_processor_result and stop_processor_result.status == "full_stop":
            stop_tokens = stop_processor_result.stop_tokens or []
            stop_string = self.tokenizer.decode(stop_tokens) if stop_tokens else ""
            generation_stop_condition = GenerationStopCondition(
                stop_reason="stop_string",
                stop_string=stop_string,
                stop_tokens=stop_tokens,
            )
        return last_segment, generation_stop_condition


# Adapted from mlx_lm.utils.server.py
# https://github.com/ml-explore/mlx-examples/blob/605c4854f1547e8eb0ef3f9c9d81c8aef3196c15/llms/mlx_lm/server.py#L43-L74
def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    tokenizer,
) -> StopStringProcessorResult:
    """Check if token generation should stop.

    Checks three conditions in order:
    1. End of sequence token from tokenizer
    2. Exact token match with stop sequences
    3. Full match between decoded text and any decoded stop sequence
    4. Partial match with any decoded stop sequence
    """
    # Check for exact token sequence matches
    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids) and tokens[-len(stop_ids) :] == stop_ids:
            return StopStringProcessorResult(
                status="full_stop", stop_tokens=stop_ids
            )

    # Check for full matches in decoded text
    tokens_str = tokenizer.decode(tokens)
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        if stop_str in tokens_str:
            return StopStringProcessorResult(
                status="full_stop", stop_tokens=stop_ids
            )

    # Check for partial matches only if no full matches were found
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        if sequence_overlap(tokens_str, stop_str):
            return StopStringProcessorResult(
                status="partial_match", stop_tokens=None
            )

    # No matches found
    return StopStringProcessorResult(status="no_match", stop_tokens=None)


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
