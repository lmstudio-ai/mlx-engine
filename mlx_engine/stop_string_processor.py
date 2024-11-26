"""Module for processing and handling stop sequences in token generation."""

from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple
import sys


StopProcessorStatus = Literal["full_stop", "partial_match", "no_match"]

class StopStringProcessorResult(NamedTuple):
    """Result of stop string processing containing status and details."""
    status: StopProcessorStatus
    # the ids that make up the stop string that was found
    stop_tokens: Optional[List[int]]
    # the decoded stop_tokens
    stop_string: Optional[str] = None

# TODO(matt): pick the right name for this - is it strings or token id? Can it be simplified?
class StopStringProcessor:
    """State-fully processes tokens to check for stop sequences during generation."""

    def __init__(self, tokenizer, stop_sequences: List[List[int]]):
        """
        Args:
            tokenizer: Tokenizer instance for encoding/decoding text
            stop_sequences: List of token sequences that signal generation should stop
        """
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.tokens_to_check_for_stopping = []

    def process_token(self, tokens: int) -> StopStringProcessorResult:
        """Process a new token and check if generation should stop.

        Args:
            tokens: New token to process

        Returns:
            StopProcessorResult indicating the state of stop string detection
        """
        self.tokens_to_check_for_stopping.append(tokens)
        
        result = stopping_criteria(
            self.tokens_to_check_for_stopping,
            self.stop_sequences,
            self.tokenizer,
        )
        
        if result.status == "no_match":
            # Clear buffer since no potential stop sequence detected
            self.tokens_to_check_for_stopping = []
            return StopStringProcessorResult(
                status="no_match",
                stop_tokens=None,
                stop_string=None
            )
            
        elif result.status == "partial_match":
            return StopStringProcessorResult(
                status="partial_match",
                stop_tokens=None,
                stop_string=None
            )
        
        elif result.status == "full_stop":
            return StopStringProcessorResult(
                status="full_stop",
                stop_tokens=result.stop_tokens,
                stop_string=self.tokenizer.decode(result.stop_tokens)
            )
        
        else:
            raise ValueError(f"Unknown StopProcessorStatus: {result.status}")

# State-less function to see how a given state of token generation relates to stop sequences
# Adapted from mlx_lm.utils.server.py
# https://github.com/ml-explore/mlx-examples/blob/605c4854f1547e8eb0ef3f9c9d81c8aef3196c15/llms/mlx_lm/server.py#L43-L74
def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    tokenizer
) -> StopStringProcessorResult:
    """Check if token generation should stop.
    
    Args:
        tokens: Current sequence of token IDs
        stop_id_sequences: List of token ID sequences that signal stopping
        tokenizer: Tokenizer for encoding/decoding text
        
    Returns:
        StopStringProcessorResult indicating match status and matched tokens
    
    Checks three stopping conditions in priority order:
    1. Exact token sequence match
    2. Full text match after decoding
    3. Partial text match after decoding
    """
    decoded_tokens = tokenizer.decode(tokens)
    
    result = (
        check_exact_token_match(tokens, stop_id_sequences) or
        check_full_text_match(decoded_tokens, stop_id_sequences, tokenizer) or
        check_partial_text_match(decoded_tokens, stop_id_sequences, tokenizer) or
        StopStringProcessorResult(status="no_match", stop_tokens=None)
    )
    
    return result

# Helpers for stopping_criteria

def check_exact_token_match(
    tokens: List[int], 
    stop_id_sequences: List[List[int]]
) -> Optional[StopStringProcessorResult]:
    """Check if tokens end with any stop sequence exactly."""
    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids) and tokens[-len(stop_ids):] == stop_ids:
            return StopStringProcessorResult(
                status="full_stop",
                stop_tokens=stop_ids
            )
    return None

def check_full_text_match(
    decoded_tokens: str,
    stop_id_sequences: List[List[int]], 
    tokenizer
) -> Optional[StopStringProcessorResult]:
    """Find earliest full text match of any stop sequence."""
    earliest_match = {
        'position': float('inf'),
        'stop_ids': None
    }
    
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        position = decoded_tokens.find(stop_str)
        
        if position != -1 and position < earliest_match['position']:
            earliest_match.update({
                'position': position,
                'stop_ids': stop_ids
            })
            
    if earliest_match['stop_ids'] is not None:
        return StopStringProcessorResult(
            status="full_stop",
            stop_tokens=earliest_match['stop_ids']
        )
    return None

def check_partial_text_match(
    decoded_tokens: str,
    stop_id_sequences: List[List[int]],
    tokenizer
) -> Optional[StopStringProcessorResult]:
    """Check for partial matches with any stop sequence."""
    for stop_ids in stop_id_sequences:
        stop_str = tokenizer.decode(stop_ids)
        if sequence_overlap(decoded_tokens, stop_str):
            return StopStringProcessorResult(
                status="partial_match",
                stop_tokens=None
            )
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
