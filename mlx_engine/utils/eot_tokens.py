from typing import Optional
from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit

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


def _get_eot_token_ids(tokenizer, model_type: Optional[str] = None) -> set[int]:
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


def sanitize_eos_tokens(model_kit: ModelKit | VisionModelKit | BatchedModelKit) -> None:
    # Remove (probably) incorrect EOS tokens
    tokenizer = model_kit.tokenizer
    temp_tokens = set()
    for id in tokenizer.eos_token_ids:
        text = tokenizer.decode(id)
        # Specific override for RNJ-1
        if model_kit.model_type == "gemma3_text" and id == 1 and text == '"':
            continue
        temp_tokens.add(id)
    temp_tokens = temp_tokens.union(_get_eot_token_ids(tokenizer, model_kit.model_type))

    if len(temp_tokens) == 0:
        raise RuntimeError(
            f"EOS tokens cannot be empty. Before cleaning, the tokens were {tokenizer.eos_token_ids}"
        )
    tokenizer.eos_token_ids = temp_tokens

    if tokenizer.eos_token_id not in tokenizer.eos_token_ids:
        tokenizer.eos_token_id = min(tokenizer.eos_token_ids)
        tokenizer._tokenizer.eos_token_id = tokenizer.eos_token_id
