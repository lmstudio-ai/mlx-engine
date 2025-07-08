from typing import List, Optional, Any

from mlx_engine.logging import log_info, log_warn, log_error
from mlx_engine.cache import make_prompt_cache
from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
import mlx.core as mx
import mlx.nn as nn
import sys


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
        keep: int = 4,
    ):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (Optional[int]): Maximum size of the key-value cache.
        """
        # utilize a simple ordered list of tokens processed so far for cache invalidation checking
        self.tokens: Optional[mx.array] = None
        self.keep = keep
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size, keep)
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.verbose = verbose
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

    @staticmethod
    def _find_matching_sequence_length(
        tokens1: mx.array,
        tokens2: mx.array,
        start1: int = 0,
        start2: int = 0,
    ) -> int:
        """
        Find the length of matching token sequence between two token arrays.

        Args:
            tokens1: First token array
            start1: Starting position in first array
            tokens2: Second token array
            start2: Starting position in second array

        Returns:
            int: Length of matching sequence
        """
        # Calculate actual bounds
        max_len1 = len(tokens1) - start1
        max_len2 = len(tokens2) - start2
        min_length = int(min(max_len1, max_len2))

        # Extract subsequences to compare
        seq1 = tokens1[start1 : start1 + min_length]
        seq2 = tokens2[start2 : start2 + min_length]

        # Find first mismatch
        mask = seq1 == seq2
        return int(mx.argmax(mask == False)) if mx.any(mask == False) else min_length  # noqa E712

    def _truncate_cache(
        self,
        prompt_tokens: mx.array,
        common_prefix_len: int,
        non_prefix_reuse_min_seq_len: int = 256,
    ) -> int:
        cache_size = len(self.tokens)
        prompt_size = len(prompt_tokens)

        # start scanning from after the common prefix
        cache_head_idx = common_prefix_len
        prompt_head_idx = common_prefix_len
        total_reused = 0

        if self.verbose:
            print(
                f"Looking for non-prefix sequences of length >= {non_prefix_reuse_min_seq_len}",
                file=sys.stderr,
            )

        while cache_head_idx < cache_size and prompt_head_idx < prompt_size:
            match_length = self._find_matching_sequence_length(
                prompt_tokens, self.tokens, prompt_head_idx, cache_head_idx
            )

            if match_length < non_prefix_reuse_min_seq_len:
                # sequence too short - advance cache pointer to find next potential match
                cache_head_idx += 1
            else:
                if self.verbose:
                    print(f"Reusing {match_length} tokens from cache", file=sys.stderr)
                print(f"idx {prompt_head_idx} {cache_head_idx}")

                # found reusable sequence - shift cache content
                for cache in self.cache:
                    cache.reuse_section(
                        prompt_head_idx, cache_head_idx, match_length
                    )

                # update the tokens to reflect the reused sequence
                for i in range(match_length):
                    self.tokens[prompt_head_idx + i] = self.tokens[cache_head_idx + i]

                # advance pointers
                cache_head_idx += match_length
                prompt_head_idx += match_length
                total_reused += match_length

        for cache in self.cache:
            cache.do_reuse()
        self.tokens = self.tokens[: common_prefix_len + total_reused]

        return total_reused

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
        common_prefix = self._find_matching_sequence_length(
            self.tokens,
            prompt_tokens,
        )

        # do reuse but only if the cache has it
        if hasattr(self.cache[0], "reuse_section"):
            n_reused_tokens = self._truncate_cache(
                prompt_tokens,
                common_prefix,
            )
            if n_reused_tokens > 0:
                log_info(
                    prefix="CacheWrapper",
                    message=f"Reused {n_reused_tokens} tokens from the cache",
                )
                common_prefix += n_reused_tokens

        # exclude some tokens from end, e.g. for kicking off generation
        if common_prefix >= len(prompt_tokens) - num_tokens_to_exclude:
            common_prefix = len(prompt_tokens) - num_tokens_to_exclude

        # Trim the cache if the common prefix is shorter than the current cache
        num_tokens_to_trim = self.cache[0].offset - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                log_warn(
                    prefix="CacheWrapper",
                    message=f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: "
                    "Cache is not trimmable. Clearing the cache instead.",
                )
                self.cache = make_prompt_cache(
                    self.model, self.max_kv_size, keep=self.keep
                )
                self.tokens = prompt_tokens
                return self.tokens
            tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
            if tokens_trimmed != num_tokens_to_trim:
                # If we trimmed fewer tokens than expected, the cache is invalid
                log_error(
                    prefix="CacheWrapper",
                    message=f"Tokens trimmed from cache ({tokens_trimmed}) is less than expected "
                    " ({num_tokens_to_trim}). Clearing the cache.",
                )
                self.cache = make_prompt_cache(
                    self.model, self.max_kv_size, keep=self.keep
                )
                self.tokens = prompt_tokens
                return self.tokens
            log_info(
                prefix="CacheWrapper",
                message=f"Trimmed {num_tokens_to_trim} tokens from the prompt cache",
            )

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
        progress_callback,
        start_progress: float,
        end_progress: float,
        chunk_size: int = 512,
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
            current_chunk_size = min(chunk_size, remaining_tokens.size)
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
            progress_callback(progress)
            mx.clear_cache()

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
            log_info(
                prefix="CacheWrapper",
                message="Disabling max_kv_size when setting a draft model for cache",
            )
            self.max_kv_size = None

        if self.draft_model is draft_model:
            # Skip if the exact same draft model instance is already in cache
            return

        # clear the current cache, append draft model cache to the end of the main model cache as per
        # https://github.com/ml-explore/mlx-examples/blob/514502da22f0dc4c1ac439bdf78c07d5ec41acf7/llms/mlx_lm/utils.py#L381-L382
        log_info(
            prefix="CacheWrapper",
            message="Clearing current prompt cache and adding draft model to the cache",
        )
        self.tokens = None
        self.cache: List[Any] = make_prompt_cache(self.model, keep=self.keep)
        if draft_model is not None:
            self.cache += make_prompt_cache(draft_model, keep=self.keep)
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
        keep: int = 4,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            prompt_progress_callback (Callable): A callback function to report prompt processing progress.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.
            keep (int): The number of tokens to always keep in the prefix of the prompt cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        if prompt_progress_callback is None:

            def prompt_progress_callback(x):
                return None

        # update keep tracking
        self.keep = keep
        for cache in self.cache:
            if hasattr(cache, "set_keep"):
                cache.set_keep(keep)

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

        Also loop when the cache does so that we accurately track what's in cache.
        """
        # this behavior is common to rolling window (n_keep = 0) and truncate middle
        # (n_keep > 0), and we should never get here with stop at max
        if len(self.tokens) >= self.max_kv_size:
            self.tokens = mx.concat(
                [self.tokens[: self.keep], self.tokens[self.keep + 1 :]]
            )
        self.tokens = mx.concat([self.tokens, mx.array([token])])
