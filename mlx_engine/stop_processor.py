from typing import List, Literal, NamedTuple, Optional, Sequence


StopReason = Literal["eos_token", "stop_string"]

StopProcessorStatus = Literal["full_stop", "partial_match", "no_match"]

class StopProcessorResult(NamedTuple):
    status: StopProcessorStatus
    stop_reason: Optional[StopReason]
    stop_tokens: Optional[List[int]]

class StopProcessor:
    def __init__(self, tokenizer, stop_sequences: List[List[int]]):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.tokens_to_check_for_stopping = []
        self.stop_sequence_suffix = None

    def process_token(self, tokens: int) -> StopProcessorResult:
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
                    self.tokens_to_check_for_stopping[-trim_length :]
                )
            return StopProcessorResult("full_stop",
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
    # TODO(matt): fix documentation
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    # first check sequences in token form
    if tokens and tokens[-1] == tokenizer.eos_token_id:
        return StopProcessorResult(status="full_stop", stop_reason="eos_token", stop_tokens=[tokenizer.eos_token_id])

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopProcessorResult(status="full_stop", stop_reason="stop_string", stop_tokens=stop_ids, is_partial_match=False)

    # must also check sequences in string form to catch inter-token stop strings
    tokens_str = tokenizer.decode(tokens)
    for stop_ids in stop_id_sequences:
        stop_ids_str = tokenizer.decode(stop_ids)
        if stop_ids_str in tokens_str:
            return StopProcessorResult(status="full_stop", stop_reason="stop_string", stop_tokens=stop_ids)

    # TODO(need to set is_partial_match to True if any stop sequence is a substring of the tokens)
    # must use STRING overlap, not token overlap
    status = "no_match"
    for stop_ids in stop_id_sequences:
        stop_ids_str = tokenizer.decode(stop_ids)
        if sequence_overlap(tokens_str, stop_ids_str):
            status = "partial_match"
            break

    return StopProcessorResult(status=status, stop_reason=None, stop_tokens=None)

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
