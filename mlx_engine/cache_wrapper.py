from typing import List, Optional, Any
import logging
from mlx_lm.models.cache import (
    make_prompt_cache,
    trim_prompt_cache,
    can_trim_prompt_cache,
)
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
import mlx.core as mx
import mlx.nn as nn
import sys
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter


PROMPT_PROCESSING_CHUNK_SIZE = 512

logger = logging.getLogger(__name__)


class StopPromptProcessing(Exception):
    """
    Exception to signal that the user aborted generation during prompt processing.
    """


class CacheWrapper:
    """
    Wrapper class for the MLX LM cache to maintain an in-memory cache
    """

    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        verbose: bool = False,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int = PROMPT_PROCESSING_CHUNK_SIZE,
    ):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (Optional[int]): Maximum size of the key-value cache.
        """
        # utilize a simple ordered list of tokens processed so far for cache invalidation checking
        self.tokens: Optional[mx.array] = None
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size)
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.verbose = verbose
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.chunk_size = chunk_size

    def _get_num_tokens_in_cache(self) -> int | None:
        """
        Get the number of tokens in the cache.

        Returns:
            int | None: The number of tokens in the cache, or None if the size cannot be determined.
        """
        for c in self.cache:
            if hasattr(c, "offset"):
                return c.offset
        return None

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
        prompt_tokens = prompt_tokens
        current_tokens = current_tokens
        # Find the minimum length between the two arrays
        min_length = min(len(current_tokens), len(prompt_tokens))

        # Compare elements up to the minimum length
        mask = prompt_tokens[:min_length] == current_tokens[:min_length]

        # Find the index where the first mismatch occurs
        if mx.any(mask == False):  # noqa E712
            common_length = int(mx.argmax(mask == False))  # noqa E712
        else:
            common_length = int(min_length)

        # Ensure that the prompt is at least num_tokens_to_exclude long
        uncached_prompt_tokens_length = len(prompt_tokens[common_length:])
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

        # Find common KV between the last generation and the current prompt
        common_prefix = self._find_common_prefix(
            self.tokens, prompt_tokens, num_tokens_to_exclude
        )

        # Trim the cache if the common prefix is shorter than the current cache
        num_tokens_in_cache = self._get_num_tokens_in_cache()
        if num_tokens_in_cache is None:
            logger.warning(
                "Could not determine the number of tokens in the cache, clearing the cache."
            )
            self.cache = make_prompt_cache(self.model, self.max_kv_size)
            self.tokens = prompt_tokens
            return self.tokens
        num_tokens_to_trim = num_tokens_in_cache - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                logger.warning(
                    f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: Cache is not trimmable. Clearing the cache instead."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
            if tokens_trimmed != num_tokens_to_trim:
                # If we trimmed fewer tokens than expected, the cache is invalid
                logger.error(
                    f"Tokens trimmed from cache ({tokens_trimmed}) is less than expected ({num_tokens_to_trim}). Clearing the cache."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            logger.info(f"Trimmed {num_tokens_to_trim} tokens from the prompt cache")

        # Keep track of the prompt tokens
        self.tokens = prompt_tokens

        if self.verbose:
            print(f"Common prefix length: {common_prefix}", file=sys.stderr)
            print(f"Trimmed tokens: {num_tokens_to_trim}", file=sys.stderr)

        # All of the common tokens are now in the cache, so we can return the remaining tokens that still need to be processed
        return prompt_tokens[common_prefix:]

    def _prefill(
        self,
        model,
        cache,
        tokens,
        reporter: PromptProgressReporter,
        is_draft: bool,
    ):
        """
        Fill a KV cache for a specific model

        Args:
            model: The model to use for cache filling
            cache: The cache to fill
            tokens: Tokens to process
            reporter: Reporter for reporting progress
            is_draft: Whether this is draft model prefill (True) or main model (False)
        """
        remaining_tokens = tokens
        num_processed = 0

        while remaining_tokens.size > 0:
            current_chunk_size = min(self.chunk_size, remaining_tokens.size)
            current_chunk = remaining_tokens[:current_chunk_size]

            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self.kv_cache_qtn_params)
            mx.eval([c.state for c in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            mx.clear_cache()

            # Report progress
            should_continue = reporter.update(is_draft, num_processed)
            if not should_continue:
                logger.info("Prompt processing was cancelled by the user.")
                num_tokens_in_cache = self._get_num_tokens_in_cache()
                if num_tokens_in_cache is not None and num_tokens_in_cache > len(
                    self.tokens
                ):
                    logger.warning(
                        "The number of tokens in the cache is greater than the number of prompt tokens. This is unexpected. Clearing the cache."
                    )
                    num_tokens_in_cache = None
                if num_tokens_in_cache is None:
                    self.cache = make_prompt_cache(self.model, self.max_kv_size)
                    self.tokens = None
                else:
                    # Remember which tokens were processed so far, so that we can continue processing at a later point
                    self.tokens = self.tokens[:num_tokens_in_cache]
                raise StopPromptProcessing

    def set_draft_model(self, draft_model: nn.Module):
        """
        Sets or updates the draft model to use in the cache.

        If the provided draft_model is already set, returns without changes.
        Otherwise, clears existing cache and rebuilds it by combining caches
        from the main model and draft model. Requires a main model to be set first.
        Args:
            draft_model: The draft model to cache. Pass None to remove draft model.

        Raises:
            ValueError: If main model hasn't been set yet.
        """
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self.max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self.max_kv_size = None

        if self.draft_model is draft_model:
            # Skip if the exact same draft model instance is already in cache
            return

        # clear the current cache, append draft model cache to the end of the main model cache as per
        # https://github.com/ml-explore/mlx-examples/blob/514502da22f0dc4c1ac439bdf78c07d5ec41acf7/llms/mlx_lm/utils.py#L381-L382
        logger.info("Clearing current prompt cache and adding draft model to the cache")
        self.tokens = None
        self.cache: List[Any] = make_prompt_cache(self.model)
        if draft_model is not None:
            self.cache += make_prompt_cache(draft_model)
        self.draft_model = draft_model

    def unset_draft_model(self):
        """Removes the draft model from the cache if one exists."""
        if self.draft_model is None:
            return
        self.draft_model = None
        self.cache = self.cache[: len(self.model.layers)]

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            reporter: Reporter for reporting prompt processing progress.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        total_prompt_tokens = len(prompt_tokens)
        prompt_tokens = self._get_unprocessed_tokens(
            prompt_tokens, num_tokens_to_exclude
        )
        cached_tokens = total_prompt_tokens - len(prompt_tokens)

        # Report begin
        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Prefill the cache with the non-excluded prompt tokens
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(prompt_tokens))
        prefill_tokens = prompt_tokens[:-num_tokens_to_exclude]

        with mx.stream(generation_stream):
            if self.draft_model is not None:
                # Fill draft model cache
                draft_cache = self.cache[len(self.model.layers) :]
                self._prefill(
                    model=self.draft_model,
                    cache=draft_cache,
                    tokens=prefill_tokens,
                    reporter=reporter,
                    is_draft=True,
                )
            # Fill main model cache
            main_cache = self.cache[: len(self.model.layers)]
            self._prefill(
                model=self.model,
                cache=main_cache,
                tokens=prefill_tokens,
                reporter=reporter,
                is_draft=False,
            )

        # Report finish
        reporter.finish(is_draft=False)

        # Return the tokens that must still be processed outside of the cache
        non_prefill_tokens = prompt_tokens[-num_tokens_to_exclude:]
        return non_prefill_tokens

    def record_generated_token(self, token):
        """
        Add the generated token to the token list, so that we can map the token to the KV cache.
        """
        self.tokens = mx.concat([self.tokens, mx.array([token])])
