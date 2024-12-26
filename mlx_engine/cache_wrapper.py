from typing import List, Optional, Any

from mlx_lm.models.cache import (
    make_prompt_cache,
    trim_prompt_cache,
    can_trim_prompt_cache,
)
from mlx_lm.utils import generation_stream
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys


class CacheWrapper:
    """
    Wrapper class for the MLX LM cache to maintain an in-memory cache
    """

    def __init__(self, model: nn.Module, max_kv_size: int, verbose: bool = False):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (int): Maximum size of the key-value cache.
        """
        # utilize a simple ordered list of tokens processed so far for cache invalidation checking
        self.tokens: Optional[mx.array] = None
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size)
        self.model = model
        self.max_kv_size = max_kv_size
        self.verbose = verbose

    @staticmethod
    def _find_common_prefix(
        current_tokens: mx.array, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ) -> int:
        """
        Determine the common prefix length between the current tokens and the prompt tokens.

        Args:
            current_tokens (mx.array): The cached tokens (self.tokens).
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The minimum length of the remaining prompt tokens array.

        Returns:
            int: The length of the common prefix.
        """
        prompt_tokens_np = prompt_tokens
        current_tokens_np = current_tokens
        # Find the minimum length between the two arrays
        min_length = min(len(current_tokens_np), len(prompt_tokens_np))

        # Compare elements up to the minimum length
        mask = prompt_tokens_np[:min_length] == current_tokens_np[:min_length]

        # Find the index where the first mismatch occurs
        if mx.any(mask == False):
            common_length = int(mx.argmax(mask == False))
        else:
            common_length = int(min_length)

        # Ensure that the prompt is at least num_tokens_to_exclude long
        uncached_prompt_tokens_length = len(prompt_tokens_np[common_length:])
        length_adjustment = max(
            0, num_tokens_to_exclude - uncached_prompt_tokens_length
        )
        common_length = max(common_length - length_adjustment, 0)
        return common_length

    def _get_unprocessed_tokens(
        self, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ):
        """
        Get the unprocessed tokens from the prompt.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The unprocessed tokens.
        """
        if self.tokens is None:
            self.tokens = prompt_tokens
            return self.tokens

        if not can_trim_prompt_cache(self.cache):
            self.cache = make_prompt_cache(self.model, self.max_kv_size)
            self.tokens = prompt_tokens
            return self.tokens

        # Find common KV between the last generation and the current prompt
        common_prefix = self._find_common_prefix(
            self.tokens, prompt_tokens, num_tokens_to_exclude
        )

        # Trim the cache history from its end so that it forgets tokens that are not in this prompt
        num_tokens_to_trim = self.cache[0].offset - common_prefix
        tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
        if tokens_trimmed != num_tokens_to_trim:
            # If we trimmed fewer tokens than expected, the cache is invalid
            self.cache = make_prompt_cache(self.model, self.max_kv_size)
            self.tokens = prompt_tokens
            return self.tokens

        # Keep track of the prompt tokens
        self.tokens = prompt_tokens

        if self.verbose:
            print(f"Common prefix length: {common_prefix}", file=sys.stderr)
            print(f"Trimmed tokens: {num_tokens_to_trim}", file=sys.stderr)

        # All of the common tokens are now in the cache, so we can return the remaining tokens that still need to be processed
        return prompt_tokens[common_prefix:]

    def update_cache(
        self,
        prompt_tokens: mx.array,
        prompt_progress_callback,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        if prompt_progress_callback is None:
            prompt_progress_callback = lambda x: None

        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        prompt_tokens = self._get_unprocessed_tokens(
            prompt_tokens, num_tokens_to_exclude
        )

        # Prefill the cache with the non-excluded prompt tokens
        prompt_progress_callback(0)
        prefill_tokens = prompt_tokens[:-num_tokens_to_exclude]
        num_total_prefill_tokens = len(prefill_tokens)
        processed: int = 0
        chunk_default_size: int = 512
        with mx.stream(generation_stream):
            while prefill_tokens.size > 0:
                chunk_size = min(chunk_default_size, prefill_tokens.size)
                chunk = prefill_tokens[:chunk_size]

                self.model(chunk[None], cache=self.cache)
                mx.eval([c.state for c in self.cache])

                prefill_tokens = prefill_tokens[chunk_size:]
                processed += chunk_size
                prompt_progress_callback((processed / num_total_prefill_tokens) * 100)
                mx.metal.clear_cache()

        # Return the tokens that must still be processed outside of the cache
        non_prefill_tokens = prompt_tokens[-num_tokens_to_exclude:]
        return non_prefill_tokens

    def record_generated_token(self, token):
        """
        Add the generated token to the token list, so that we can map the token to the KV cache.
        """
        self.tokens = mx.concat([self.tokens, mx.array([token])])
