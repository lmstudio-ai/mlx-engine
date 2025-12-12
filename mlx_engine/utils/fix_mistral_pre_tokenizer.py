from tokenizers import Tokenizer
from pathlib import Path
from mlx_lm.tokenizer_utils import TokenizerWrapper
import logging
from transformers import LlamaTokenizer
import traceback

logger = logging.getLogger(__name__)

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

    # Fix pre-tokenizer
    try:
        tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        tokenizer._tokenizer._tokenizer.pre_tokenizer = tok.pre_tokenizer
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
