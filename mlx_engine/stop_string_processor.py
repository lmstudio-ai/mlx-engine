"""Module for processing and handling stop strings during token generation."""

from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple

StopStringProcessorStatus = Literal["full_stop", "partial_match", "no_match"]


class StopStringProcessorResult(NamedTuple):
    """Result of stop string processing containing status and details."""

    status: StopStringProcessorStatus
    stop_string: Optional[str] = None  # populated if status is "full_stop"
    stop_tokens: Optional[List[int]] = None  # populated if status is "full_stop"


class StopStringProcessor:
    """State-fully processes tokens to check for stop strings during generation."""

    def __init__(self, stop_strings: List[str], tokenizer):
        """
        Args:
            stop_strings: List of strings that should stop generation if found
            tokenizer: Tokenizer instance for encoding token IDs to text
        """
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.token_id_buffer = []

    def process_token(self, tokens: int) -> StopStringProcessorResult:
        """Process a new token and check if generation should stop.

        Args:
            tokens: New token to process

        Returns:
            StopProcessorResult indicating the state of stop string detection
        """
        if len(self.stop_strings) == 0:
            return StopStringProcessorResult(
                status="no_match", stop_string=None, stop_tokens=None
            )

        self.token_id_buffer.append(tokens)

        result = stopping_criteria(
            string=self.tokenizer.decode(self.token_id_buffer),
            stop_strings=self.stop_strings,
        )

        if result.status == "no_match":
            # Can clear the buffer in no partial or full matches with stop sequences
            self.token_id_buffer = []
            return StopStringProcessorResult(
                status="no_match", stop_string=None, stop_tokens=None
            )

        elif result.status == "partial_match":
            return StopStringProcessorResult(
                status="partial_match", stop_string=None, stop_tokens=None
            )

        elif result.status == "full_stop":
            return StopStringProcessorResult(
                status="full_stop",
                stop_string=result.stop_string,
                stop_tokens=self.token_id_buffer,
            )

        else:
            raise ValueError(f"Unknown StopProcessorStatus: {result.status}")


class StoppingCriteriaResult(NamedTuple):
    status: StopStringProcessorStatus
    stop_string: Optional[str] = None  # populated if status is "full_stop"


def stopping_criteria(
    string: str,
    stop_strings: List[str],
) -> StoppingCriteriaResult:
    """Check if stop_strings match or partially match the string.

    Args:
        string: The string to check for stop strings
        stop_strings: List of strings that should stop generation if found

    Returns:
        StopStringProcessorResult indicating match status, and stop string if matched

    Checks two stopping conditions in priority order:
    1. Exact stop string match
    2. Partial stop string match
    """

    result = (
        check_full_text_match(string, stop_strings)
        or check_partial_text_match(string, stop_strings)
        or StoppingCriteriaResult(status="no_match", stop_string=None)
    )

    return result


# Helpers for stopping_criteria


def check_full_text_match(
    string: str, stop_strings: List[str]
) -> Optional[StoppingCriteriaResult]:
    """Find earliest full text match of any stop sequence."""
    earliest_match = {"position": float("inf"), "stop_string": None}

    for stop_string in stop_strings:
        position = string.find(stop_string)

        if position != -1 and position < earliest_match["position"]:
            earliest_match.update({"position": position, "stop_string": stop_string})

    if earliest_match["stop_string"] is not None:
        return StoppingCriteriaResult(
            status="full_stop", stop_string=earliest_match["stop_string"]
        )
    return None


def check_partial_text_match(
    string: str, stop_strings: List[str]
) -> Optional[StopStringProcessorResult]:
    """Check for partial matches with any stop sequence."""
    for stop_string in stop_strings:
        if sequence_overlap(string, stop_string):
            return StopStringProcessorResult(status="partial_match", stop_string=None)
    return None


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
