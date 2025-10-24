from typing import Callable, List, Optional, Any
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

        # Vision prompt caching state
        self.prev_images_hash: Optional[str] = None
        self.prev_raw_prompt_tokens: Optional[List[int]] = None

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
            self.clear_vision_cache()
            return self.tokens
        num_tokens_to_trim = num_tokens_in_cache - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                logger.warning(
                    f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: Cache is not trimmable. Clearing the cache instead."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                self.clear_vision_cache()
                return self.tokens
            tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
            if tokens_trimmed != num_tokens_to_trim:
                # If we trimmed fewer tokens than expected, the cache is invalid
                logger.error(
                    f"Tokens trimmed from cache ({tokens_trimmed}) is less than expected ({num_tokens_to_trim}). Clearing the cache."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                self.clear_vision_cache()
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
        progress_callback: Callable[[float], bool],
        start_progress: float,
        end_progress: float,
    ):
        """
        Fill a KV cache for a specific model

        Args:
            model: The model to use for cache filling
            cache: The cache to fill
            tokens: Tokens to process
            progress_callback: Callback for reporting progress
            start_progress: Starting progress percentage
            end_progress: Ending progress percentage
        """
        remaining_tokens = tokens
        num_processed = 0
        total_tokens = len(tokens)

        while remaining_tokens.size > 0:
            current_chunk_size = min(self.chunk_size, remaining_tokens.size)
            current_chunk = remaining_tokens[:current_chunk_size]

            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self.kv_cache_qtn_params)
            mx.eval([c.state for c in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            # Scale progress to fit between start_progress and end_progress
            progress = start_progress + (end_progress - start_progress) * (
                num_processed / total_tokens
            )
            mx.clear_cache()
            should_continue = progress_callback(progress)
            if should_continue is False:  # If it's None, assume continue generation
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
                    self.clear_vision_cache()
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
        self.clear_vision_cache()
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
        prompt_progress_callback,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            prompt_progress_callback (Callable): A callback function to report prompt processing progress.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        if prompt_progress_callback is None:

            def prompt_progress_callback(_) -> bool:
                return True

        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        prompt_tokens = self._get_unprocessed_tokens(
            prompt_tokens, num_tokens_to_exclude
        )

        # Prefill the cache with the non-excluded prompt tokens
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(prompt_tokens))
        prefill_tokens = prompt_tokens[:-num_tokens_to_exclude]
        prompt_progress_callback(0)
        with mx.stream(generation_stream):
            if self.draft_model is not None:
                # Fill draft model cache (0% to 50% progress)
                draft_cache = self.cache[len(self.model.layers) :]
                self._prefill(
                    model=self.draft_model,
                    cache=draft_cache,
                    tokens=prefill_tokens,
                    progress_callback=prompt_progress_callback,
                    start_progress=0,
                    end_progress=50,
                )
            # Fill main model cache (50% to 100% progress for draft model, 0% to 100% otherwise)
            main_cache = self.cache[: len(self.model.layers)]
            self._prefill(
                model=self.model,
                cache=main_cache,
                tokens=prefill_tokens,
                progress_callback=prompt_progress_callback,
                start_progress=50 if self.draft_model is not None else 0,
                end_progress=100,
            )

        # Return the tokens that must still be processed outside of the cache
        non_prefill_tokens = prompt_tokens[-num_tokens_to_exclude:]
        return non_prefill_tokens

    def record_generated_token(self, token):
        """
        Add the generated token to the token list, so that we can map the token to the KV cache.
        """
        self.tokens = mx.concat([self.tokens, mx.array([token])])

    def _compute_images_hash(self, images_b64: List[str]) -> str:
        """Compute hash of images for cache validation."""
        import hashlib

        combined = "".join(images_b64)
        return hashlib.sha256(combined.encode()).hexdigest()

    def can_reuse_vision_cache(
        self, images_b64: List[str], raw_prompt_tokens: List[int]
    ) -> bool:
        """
        Check if we can skip expensive vision processing and reuse cached KV states.

        Args:
            images_b64: Current request's base64-encoded images
            raw_prompt_tokens: Current request's raw prompt tokens (before vision processing)

        Returns:
            bool: True if we can skip vision processing, False otherwise
        """
        if self.prev_images_hash is None or self.prev_raw_prompt_tokens is None:
            return False

        # Check if images are identical
        current_images_hash = self._compute_images_hash(images_b64)
        if current_images_hash != self.prev_images_hash:
            return False

        # Check if current prompt extends previous prompt
        if len(raw_prompt_tokens) <= len(self.prev_raw_prompt_tokens):
            return False

        # Check if prefix matches exactly
        if (
            raw_prompt_tokens[: len(self.prev_raw_prompt_tokens)]
            != self.prev_raw_prompt_tokens
        ):
            return False

        return True

    def record_vision_state(self, images_b64: List[str], raw_prompt_tokens: List[int]):
        """
        Record vision processing state for future cache validation.

        Args:
            images_b64: Base64-encoded images that were processed
            raw_prompt_tokens: Raw prompt tokens (before vision processing) that were used
        """
        self.prev_images_hash = self._compute_images_hash(images_b64)
        self.prev_raw_prompt_tokens = raw_prompt_tokens

    def clear_vision_cache(self):
        """Clear vision-specific cache state."""
        self.prev_images_hash = None
        self.prev_raw_prompt_tokens = None
