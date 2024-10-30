from typing import List, Literal, NamedTuple, Optional, Sequence, Union

StopReason = Literal["eos_token", "stop_string"]

class StoppingCriteriaResult(NamedTuple):
    stop_reason: Optional[StopReason]
    stop_tokens: Optional[List[int]]
    is_partial_match: bool

# Adapted from mlx_lm.utils.server.py
# https://github.com/ml-explore/mlx-examples/blob/605c4854f1547e8eb0ef3f9c9d81c8aef3196c15/llms/mlx_lm/server.py#L43-L74
def stopping_criteria(
        tokens: List[int],
        stop_id_sequences: List[List[int]],
        tokenizer,
) -> StoppingCriteriaResult:
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
    # first try to cheaply check if tokens can be used to stop
    if tokens and tokens[-1] == tokenizer.eos_token_id:
        return StoppingCriteriaResult(stop_reason="eos_token", stop_tokens=[tokenizer.eos_token_id], is_partial_match=False)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StoppingCriteriaResult(stop_reason="stop_string", stop_tokens=stop_ids, is_partial_match=False)

    # I think we also need to detokenize the tokens, and detokenize the stop_id_sequences
    # in order to find if there are any stop_id_sequence strings that are substring of the
    # detokenized tokens

    # This severely slows down the code once the number of tokens is large
    # we need a way for this to be stateful
    tokens_str = tokenizer.decode(tokens)
    for stop_ids in stop_id_sequences:
        stop_ids_str = tokenizer.decode(stop_ids)
        if stop_ids_str in tokens_str:
            return StoppingCriteriaResult(stop_reason="stop_string", stop_tokens=stop_ids, is_partial_match=True)

    # TODO(need to set is_partial_match to True if any stop sequence is a substring of the tokens)
    # must use STRING overlap, not token overlap
    is_partial_match = False
    for stop_ids in stop_id_sequences:
        stop_ids_str = tokenizer.decode(stop_ids)
        if sequence_overlap(tokens_str, stop_ids_str):
            is_partial_match = True
            break

    return StoppingCriteriaResult(stop_reason=None, stop_tokens=None, is_partial_match=is_partial_match)

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
