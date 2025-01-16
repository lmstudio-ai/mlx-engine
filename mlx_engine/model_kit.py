from typing import List, Optional

import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a text model.

    Args:
        model_path (Path): Path to the model directory containing model files.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        kv_bits (Optional[int]): Number of bits for KV cache quantization. None disables quantization.
        kv_group_size (int): Group size for KV cache quantization. Defaults to 64.
        quantized_kv_start (int): Step to begin KV cache quantization when enabled. Defaults to 0.
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

    def __init__(
        self,
        model_path: Path,
        max_kv_size: Optional[int],
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        if any([kv_group_size, quantized_kv_start]) and kv_bits is None:
            raise ValueError(
                "Enabling KV Cache Quantization requires kv_bits to be set"
            )
        if (
            not any([kv_bits, kv_group_size, quantized_kv_start])
            and max_kv_size is None
        ):
            raise ValueError("Context length setting is required")

        self.model_path = model_path
        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(self.model, max_kv_size)
        self.max_kv_size = max_kv_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

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

    @property
    def language_model(self):
        return self.model
