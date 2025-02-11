"""Module for processing and handling stop strings during token generation."""

from typing import List, Literal, NamedTuple, Optional, Sequence

StopStringProcessorStatus = Literal[
    "full_stop", "partial_match", "no_match", "multi_byte"
]

REPLACEMENT_CHAR = "\ufffd"


class StopStringProcessorResult(NamedTuple):
    """Result of stop string processing containing status and details."""

    status: StopStringProcessorStatus
    stop_string: Optional[str] = None  # populated if status is "full_stop"
    # sequence of tokens that the stop_string was found in
    stop_tokens: Optional[List[int]] = None  # populated if status is "full_stop"


class StopStringProcessor:
    """State-fully processes tokens to check for stop strings during generation."""

    def __init__(self, stop_strings: List[str], tokenizer):
        """
        Args:
            stop_strings: List of strings that should stop generation if found
            tokenizer: Tokenizer instance for encoding token IDs to text

        Raises:
            ValueError: If stop_strings is empty or contains invalid values
            TypeError: If stop_strings contains non-string values
        """
        if not stop_strings:
            raise ValueError("Must provide at least one stop string")

        if not all(isinstance(s, str) for s in stop_strings):
            raise TypeError("All stop strings must be strings")

        if any(not stop_string for stop_string in stop_strings):
            raise ValueError("Stop strings cannot be empty")

        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.token_id_buffer = []

    def process_token(self, token: int) -> StopStringProcessorResult:
        """Process a new string segment and check how it relates to stop strings.

        Args:
            segment: The new string segment to process

        Returns:
            StopProcessorResult indicating the state of stop string detection
        """
        if len(self.stop_strings) == 0:
            return StopStringProcessorResult(
                status="no_match", stop_string=None, stop_tokens=None
            )

        self.token_id_buffer.append(token)

        result = self._stopping_criteria(
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

        elif result.status == "multi_byte":
            return StopStringProcessorResult(
                status="multi_byte", stop_string=None, stop_tokens=None
            )

        elif result.status == "full_stop":
            return StopStringProcessorResult(
                status="full_stop",
                stop_string=result.stop_string,
                stop_tokens=self.token_id_buffer,
            )

        else:
            raise ValueError(f"Unknown StopProcessorStatus: {result.status}")

    class _StoppingCriteriaResult(NamedTuple):
        status: StopStringProcessorStatus
        stop_string: Optional[str] = None  # populated if status is "full_stop"

    def _stopping_criteria(
        self,
        string: str,
        stop_strings: List[str],
    ) -> _StoppingCriteriaResult:
        """Check if stop strings match or partially match the input string

        Args:
            string: The string to check for stop strings
            stop_strings: List of strings that should stop generation if found

        Returns:
            StopStringProcessorResult indicating match status and stop string if matched

        Checks stopping conditions in priority order:
        1. Incomplete UTF-8 string
        2. Exact stop string match
        3. Partial stop string match
        """

        result = (
            self._check_incomplete_utf8(string)
            or self._check_full_text_match(string, stop_strings)
            or self._check_partial_text_match(string, stop_strings)
            or self._StoppingCriteriaResult(status="no_match", stop_string=None)
        )

        return result

    def _check_incomplete_utf8(self, string: str) -> Optional[_StoppingCriteriaResult]:
        if len(string) and string[-1] == REPLACEMENT_CHAR:
            return self._StoppingCriteriaResult(status="multi_byte", stop_string=None)
        return None

    def _check_full_text_match(
        self, string: str, stop_strings: List[str]
    ) -> Optional[_StoppingCriteriaResult]:
        """Find earliest full text match of any stop sequence."""
        earliest_match = {"position": float("inf"), "stop_string": None}

        for stop_string in stop_strings:
            position = string.find(stop_string)

            if position != -1 and position < earliest_match["position"]:
                earliest_match.update(
                    {"position": position, "stop_string": stop_string}
                )

        if earliest_match["stop_string"] is not None:
            return self._StoppingCriteriaResult(
                status="full_stop", stop_string=earliest_match["stop_string"]
            )
        return None

    def check_partial_token_match(
        self, token_sequence: List[int], stop_token_sequences: List[List[int]]
    ) -> Optional[_StoppingCriteriaResult]:
        """Check for partial matches with any stop sequence."""
        for stop_token_sequence in stop_token_sequences:
            if self._sequence_overlap(token_sequence, stop_token_sequence):
                return self._StoppingCriteriaResult(
                    status="partial_match", stop_string=None
                )
        return None

    def _check_partial_text_match(
        self, string: str, stop_strings: List[str]
    ) -> Optional[StopStringProcessorResult]:
        """Check for partial matches with any stop sequence."""
        for stop_string in stop_strings:
            if self._sequence_overlap(string, stop_string):
                return StopStringProcessorResult(
                    status="partial_match", stop_string=None
                )
        return None

    def _sequence_overlap(self, s1: Sequence, s2: Sequence) -> bool:
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
