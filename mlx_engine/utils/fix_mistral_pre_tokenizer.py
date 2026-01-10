from tokenizers import Tokenizer
from pathlib import Path
from mlx_lm.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
import logging
from transformers import LlamaTokenizer
import traceback

logger = logging.getLogger(__name__)

# List taken from here
# https://github.com/huggingface/transformers/blob/b9951b4/src/transformers/tokenization_utils_tokenizers.py#L1187
_LEGACY_MISTRAL_MODEL_TYPES = [
    "mistral",
    "mistral3",
    "voxtral",
    "ministral",
    "pixtral",
]


def fix_mistral_pre_tokenizer(
    *, tokenizer: TokenizerWrapper, model_path: Path, model_type: str
) -> None:
    """
    transformers v5 introduces breaking changes in their tokenization framework.
    Unfortunately, some of the mistral models were patched incorrectly in transformers during this breakage.

    In transformers-world, using mistral_common for tokenization is a possibility, but we can't use that here
    since mistral_common (by design) doesn't tokenize the special tokens correctly

    For mistral models that were introduced in transformers v4, check to see if tokenization is broken. The breakage
    specifically happens for LlamaTokenizer instances

    Tokenization is considered broken if it doesn't handle whitespace correctly. For example, tokenizing
    `Hello world` should result in tokens `["Hello", " world"]`, and not `["Hello", "world"]`. Note the missing
    whitespace before `world`
    
    This also fixes decoding issues by using BPEStreamingDetokenizer which properly handles
    byte-level BPE tokens like Ġ (the space marker used by Mistral/GPT-2 style tokenizers).
    """
    if model_type not in _LEGACY_MISTRAL_MODEL_TYPES:
        return
    logger.info("Detected mistral model. Checking if tokenizer needs fixing...")
    if not isinstance(tokenizer._tokenizer, LlamaTokenizer):
        logger.info(f"Tokenizer is of type {type(tokenizer._tokenizer)}. Skipping fix.")
        return
    if not _tokenizer_is_broken(tokenizer):
        logger.info("Tokenizer working as expected.")
        return

    # Fix pre-tokenizer and detokenizer
    try:
        tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        # Fix encoding (pre-tokenizer)
        tokenizer._tokenizer._tokenizer.pre_tokenizer = tok.pre_tokenizer
        
        # Fix decoding by using BPEStreamingDetokenizer instead of the default
        # NaiveStreamingDetokenizer. BPEStreamingDetokenizer properly handles
        # byte-level BPE tokens like Ġ (which represents a space).
        # This is the same detokenizer that mlx_vlm uses for Mistral models.
        # Note: TokenizerWrapper.detokenizer is a property that creates instances
        # from _detokenizer_class, so we need to override that class.
        tokenizer._detokenizer_class = BPEStreamingDetokenizer
        logger.info("Replaced detokenizer class with BPEStreamingDetokenizer")
    except Exception:
        logger.warning(f"Failed to fix tokenizer: {traceback.format_exc()}.")
        return

    if _tokenizer_is_broken(tokenizer):
        logger.warning("Tokenizer could not be fixed.")
        return

    logger.info("Successfully fixed tokenizer.")


def _tokenizer_is_broken(tokenizer: TokenizerWrapper) -> bool:
    """
    `["about", "Paris"]` shows us that the tokenization is broken because
    the whitespace is missing between `about` and `Paris`.
    """
    test_prompt = "Tell me about Paris"
    tokens = tokenizer.tokenize(test_prompt)
    return tokens[-2:] == ["about", "Paris"]

