import sys
from typing import List, Optional

from mlx_engine.logging import log_info, log_warn
import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L782
VALID_KV_BITS = (2, 3, 4, 6, 8)

# https://github.com/ml-explore/mlx/blob/f288db8d34c0bcfa0867b6458ab0277c5e86ed45/mlx/fast.cpp#L775
VALID_KV_GROUP_SIZE = (32, 64, 128)


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a text model.

    Args:
        model_path (Path): Path to the model directory containing model files.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        kv_bits (Optional[int]): Number of bits for KV cache quantization. None disables quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization. Defaults to 64.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled. Defaults to 0.
    """

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    draft_model: Optional[nn.Module] = None

    def _vocab_only_init(self, model_path: Path):
        log_info(
            prefix="ModelKit",
            message=f"Loading model (vocab-only) from {model_path}...",
        )
        self.tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(model_path)
        self.detokenizer = self.tokenizer.detokenizer
        log_info(prefix="ModelKit", message="Model (vocab-only) loaded successfully")

    def _full_model_init(
        self,
        model_path: Path,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        self._validate_kv_cache_quantization_params(
            kv_bits, kv_group_size, quantized_kv_start
        )
        if kv_bits and max_kv_size is not None:
            # Quantized KV cache is only supported for non-rotating KV cache
            log_warn(
                prefix="ModelKit",
                message="max_kv_size is ignored when using KV cache quantization",
            )
            max_kv_size = None

        self.model_path = model_path
        log_info(prefix="ModelKit", message=f"Loading model from {model_path}...")
        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(self.model, max_kv_size)
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        log_info(prefix="ModelKit", message="Model loaded successfully")

    def __init__(
        self,
        model_path: Path,
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        if vocab_only:
            self._vocab_only_init(model_path)
        else:
            self._full_model_init(
                model_path,
                max_kv_size,
                kv_bits,
                kv_group_size,
                quantized_kv_start,
            )

    @staticmethod
    def _validate_kv_cache_quantization_params(
        kv_bits: Optional[int],
        kv_group_size: Optional[int],
        quantized_kv_start: Optional[int],
    ):
        if any([kv_group_size, quantized_kv_start]) and kv_bits is None:
            raise ValueError(
                "Enabling KV Cache Quantization requires kv_bits to be set"
            )

        if kv_bits and kv_bits not in VALID_KV_BITS:
            raise ValueError(f"Invalid kv_bits value. Must be one of {VALID_KV_BITS}")
        if kv_group_size and kv_group_size not in VALID_KV_GROUP_SIZE:
            raise ValueError(
                f"Invalid kv_group_size value. Must be one of {VALID_KV_GROUP_SIZE}"
            )

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if type(ids) == int:
            return [ids]
        return ids

    def process_prompt(
        self,
        prompt_tokens,
        img_b64,
        prompt_progress_callback,
        repetition_context_size,
        generate_args,
    ) -> mx.array:
        """
        This method processes the prompt, adding its tokens to the cache history

        Call this before starting evaluation

        Returns the uncached tokens as input for the `generate_step` function, and
        updates generate_args with the cache history
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Prompt tokens must be non-empty")

        # Check for common tokens with the previous cache and re-use the cache if possible
        prompt_tokens = self.cache_wrapper.update_cache(
            mx.array(prompt_tokens),
            prompt_progress_callback,
            num_tokens_to_exclude=repetition_context_size,
        )
        generate_args["prompt_cache"] = self.cache_wrapper.cache

        return prompt_tokens

    def update_cache_wrapper(self, token: int) -> None:
        self.cache_wrapper.record_generated_token(token)

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        path = Path(path)
        if self.tokenizer is None:
            log_warn(
                prefix="ModelKit",
                message="Draft model compatibility check requires at least a vocab-only "
                "loaded main model",
            )
            return False
        draft_tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(path)
        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            return False
        return True

    def load_draft_model(self, path: str | Path) -> None:
        log_info(prefix="ModelKit", message=f"Loading draft model from {path}...")
        path = Path(path)
        if self.model is None:
            raise ValueError("Main model must be loaded before loading a draft model")
        if not self.is_draft_model_compatible(path):
            raise ValueError("Draft model is not compatible with main model")
        self.draft_model, _ = mlx_lm.utils.load(path)
        self.cache_wrapper.set_draft_model(self.draft_model)
        log_info(prefix="ModelKit", message="Draft model loaded")

    def unload_draft_model(self) -> None:
        if self.draft_model is None:
            log_info(prefix="ModelKit", message="No loaded draft model to unload")
        else:
            self.draft_model = None
            self.cache_wrapper.unset_draft_model()
        # Noticed that draft model memory would not be released without clearing metal cache
        mx.metal.clear_cache()

    @property
    def language_model(self):
        return self.model
