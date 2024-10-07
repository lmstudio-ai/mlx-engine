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

    def __init__(self, model_path, max_kv_size):
        self.model_path = Path(model_path)
        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(self.model, max_kv_size, verbose=False)

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if type(ids) == int:
            return [ids]
        return ids

    def process_prompt(
        self, prompt_tokens, img_b64, prompt_progress_callback, generate_args
    ) -> mx.array:
        """
        This method processes the prompt and adding its tokens to the cache history

        Call this before starting evaluation

        Returns the uncached tokens as input for the `generate_step` function, and
        updates generate_args with the cache history
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Prompt tokens must be non-empty")

        # prefill cache with prompt_tokens, except those that need to have a repetition penalty applied
        # (repetition penalty not currently possible for cached tokens)
        if "repetition_context_size" not in generate_args:
            generate_args["repetition_context_size"] = (
                20  # default value for mlx_lm.utils.generate_step
            )
        repetition_context_size = generate_args["repetition_context_size"]

        cache_history, generate_step_input = self.cache_wrapper.update_cache(
            prompt_tokens, 
            num_tokens_to_exclude=repetition_context_size,
            progress_callback=prompt_progress_callback,
        )

        generate_args["cache_history"] = cache_history

        return generate_step_input

    @property
    def language_model(self):
        return self.model
