import time
from typing import Callable, List, Optional, Tuple

from mlx_lm.utils import make_kv_caches, RotatingKVCache
import mlx.core as mx
import mlx.nn as nn


class CacheWrapper:
    """
    Wrapper class for the MLX cache to maintain an in-memory cache
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
        # will always be a list of RotatingKVCache objects since we pass max_kv_size
        self.cache: List[RotatingKVCache] = make_kv_caches(model, max_kv_size)
        self.model = model
        self.max_kv_size = max_kv_size
        self.verbose = verbose

    def _reset_cache_if_necessary(self, tokens: List[int]) -> bool:
        """
        Resets the cache if the incoming tokens do not match the start of the existing cache.
        Returns True if the cache was reset, False otherwise.

        This is the most naive implementation of cache invalidation.
        """
        # loop through reasons to short-circuit detailed reset checking
        if self.tokens is None:
            if self.verbose:
                print("No cached tokens, no need to reset", flush=True)
            return False

        # loop through all potential reset reasons
        y = mx.array(tokens)
        reset = False
        if len(y) == 0:
            reset = True
            reason = "Passed token list is empty"
        elif len(self.tokens) > len(y):
            reset = True
            reason = "Passed token list is shorter than cached tokens"
        elif not mx.all(y[: len(self.tokens)] == self.tokens):
            reset = True
            reason = (
                "Beginning sequence of passed token list does not match cached tokens"
            )

        if reset:
            if self.verbose:
                print(f"Resetting cache with reason: {reason}", flush=True)
            self.cache = make_kv_caches(self.model, self.max_kv_size)
            self.tokens = None
        else:
            if self.verbose:
                print("Cache is valid with passed tokens", flush=True)

        return reset

    def _determine_tokens_to_process(self, tokens: mx.array) -> mx.array:
        """
        Returns an mx.array of the tokens that should be processed given the state of the cache and
        an array of tokens.

        Throws if there isn't a contiugous sequence of tokens that can be naively processed in order
        """
        # no tokens yet, so should process all
        if self.tokens is None:
            return tokens
        # if y is shorter than self.tokens, throw an error
        elif len(tokens) < len(self.tokens):
            raise RuntimeError(
                "Passed tokens list cannot be shorter than cached tokens"
            )
        # if start of y is not the same as self.tokens, throw an error
        elif not mx.all(tokens[: len(self.tokens)] == self.tokens):
            raise RuntimeError(
                "Passed tokens list does not start with previously cached tokens, cannot safely "
                "process new tokens continguously into cache"
            )
        # figure out which tokens are new
        else:
            new_tokens = tokens[len(self.tokens) :]
            return new_tokens

    # adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/cache_prompt.py#L121-L137
    def _process_tokens_into_cache(
        self,
        tokens_to_process: mx.array,
        progress_callback: Optional[Callable[[float], None]],
        step_size: int,
    ) -> None:
        """
        Process tokens and update the cache.

        Args:
            tokens_to_process (mx.array): Tokens to be processed and added to the cache.
            progress_callback (function): Callback function to report progress.
            step_size (int): Number of tokens to process in each step.
        """
        # signal that processing has started
        if progress_callback:
            progress_callback(0)

        processed: int = 0
        start = time.time()
        while processed < len(tokens_to_process):
            chunk: mx.array = tokens_to_process[processed : processed + step_size]
            self.model(chunk[None], cache=self.cache)
            mx.eval([c.state for c in self.cache])
            self.tokens: mx.array = (
                mx.concatenate([self.tokens, chunk])
                if self.tokens is not None
                else chunk
            )
            processed += chunk.size
            speed = processed / (time.time() - start)
            percentage: float = (processed / len(tokens_to_process)) * 100
            if self.verbose:
                print(
                    f"\rProcessed {processed:d} tokens ({speed:.2f} tok/s) - {percentage:.2f}% complete",
                    flush=True,
                )
            if progress_callback:
                progress_callback(percentage)

    def update_cache(
        self, prompt_tokens: mx.array, num_tokens_to_exclude, progress_callback=None, step_size: int = 512, 
    ) -> Tuple[List[Tuple[mx.array, mx.array]], mx.array]:
        """
        Update the cache by filling it with the prompt tokens in a way that:
         1. Ensures it is valid for the prompt tokens
         2. Can be told to exclude a certain number of tokens from the end of the prompt, so that those tokens can
            be processed outside of the cache route (e.g. for repetition penalty)

        Args:
            prompt_tokens (mx.array): Tokens to sync the cache with.
            num_tokens_to_exclude (int): Number of tokens to exclude from caching.
            progress_callback (function, optional): Function to report progress.
            step_size (int, optional): Batch size for processing. Defaults to 512.

        Returns:
            tuple: Cache history object and uncached tokens.
        """
        if len(prompt_tokens) == 0:
            return prompt_tokens
        
        # first, reset the cache if the full prompt tokens do not match the start of the cache
        # we need to see the whole prompt so that if the cache is not valid for it, we can reset it
        self._reset_cache_if_necessary(prompt_tokens)
        
        tokens_to_cache = mx.array(prompt_tokens[:-num_tokens_to_exclude])
        uncached_tokens = mx.array(prompt_tokens[-num_tokens_to_exclude:])

        # second, reset the cache if the tokens to cache do not match the start of the cache
        # we need to do this so that we can safely process the uncached tokens
        self._reset_cache_if_necessary(tokens_to_cache)

        # figure out which tokens are not already in cache, and add them
        tokens_to_process = self._determine_tokens_to_process(tokens_to_cache)
        self._process_tokens_into_cache(tokens_to_process, progress_callback, step_size)

        return self._to_cache_history(), uncached_tokens

    # adapted from https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/cache_prompt.py#L140-L143
    # and https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/generate.py#L139-L149
    def _to_cache_history(self) -> List[Tuple[mx.array, mx.array]]:
        """
        Convert the current cache to a cache history format.

        Returns:
            List[Tuple[mx.array, mx.array]]: A list of tuples containing keys and values from the cache.
            Returns None if there are no tokens in the cache.
        """
        if self.tokens is None:
            return None

        cache_history = []
        for c in self.cache:
            keys = c.state[0][..., : c.offset, :]
            values = c.state[1][..., : c.offset, :]
            cache_history.append((keys, values))

        return cache_history
