from typing import Optional

# Taken from https://github.com/ggml-org/llama.cpp/blob/971f245/src/llama-vocab.cpp#L1807-L1814
DEFAULT_EOT_TOKENS = [
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<EOT>",
    "_<EOT>",
    "<｜end▁of▁sentence｜>",
]

MODEL_TYPE_TO_EOT_TOKENS = {"gpt_oss": ["<|return|>", "<|call|>"]}


def get_eot_token_ids(tokenizer, model_type: Optional[str] = None) -> set[int]:
    """
    Get the token ID of common end-of-text tokens, using the provided tokenizer.

    If the EOT token str cannot be converted into a single token ID, it is discarded as a candidate.
    """
    if (
        isinstance(model_type, str)
        and len(MODEL_TYPE_TO_EOT_TOKENS.get(model_type, [])) > 0
    ):
        eot_tokens = MODEL_TYPE_TO_EOT_TOKENS[model_type]
    else:
        eot_tokens = DEFAULT_EOT_TOKENS

    # Convert EOT tokens to token IDs
    eot_token_ids = [
        tokenizer.encode(eot_str, add_special_tokens=False) for eot_str in eot_tokens
    ]

    # Find all elements that are either a single integer or a list with a single integer
    single_int = [token_id for token_id in eot_token_ids if isinstance(token_id, int)]
    single_element_list = [
        token_id[0]
        for token_id in eot_token_ids
        if isinstance(token_id, list) and len(token_id) == 1
    ]

    return set(single_int + single_element_list)
