from typing import List, Optional

import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a text model
    """

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    max_kv_size: int = None

    def __init__(self, model_path: Path, max_kv_size: int):
        self.model_path = model_path
        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(self.model, max_kv_size)
        self.max_kv_size = max_kv_size

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
