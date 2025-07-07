from typing import List, Optional, Any

from mlx_engine.logging import log_info, log_warn, log_error
from mlx_lm.models.cache import (
    trim_prompt_cache,
    can_trim_prompt_cache,
    RotatingKVCache,
    KVCache
)
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
import mlx.core as mx
import mlx.nn as nn
import sys


# TODO(christian-lms) DO NOT HARDCODE ME (or at least move it somewhere else)
MAYBE_ATTN_NAMES = ["self_attn", "attention", "attn", "mixer", "norm_attn_norm"]
MAYBE_ROPE_NAMES = ["rope", "rotary_emb"]


def _maybe_get_rope(layer: nn.Module) -> Optional[nn.Module]:
    for maybe_rope_name in MAYBE_ROPE_NAMES:
        if hasattr(layer, maybe_rope_name):
            # found it
            return getattr(layer, maybe_rope_name)
    for maybe_attn_name in MAYBE_ATTN_NAMES:
        if hasattr(layer, maybe_attn_name):
            # move down one level
            return _maybe_get_rope(getattr(layer, maybe_attn_name))
    # no dice
    return None


def maybe_get_rope(model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
    """Attempt to find the RoPE module from a layer of an MLX-LM LLM.

    Args:
        model (nn.Module): The LLM to search for the RoPE modules of. 
        layer_idx (int): The layer of the LLM to get the RoPE module from.

    Returns:
        Optional[nn.Module]: The RoPE module if found, else None
    """
    # we can assume model has attribute layers because make_prompt_cache does
    if layer_idx > len(model.layers):
        # TODO(christian-lms): fail silently or throw here?
        return None
    layer = model.layers[layer_idx]
    if not isinstance(layer, nn.Module):
        return None
    return _maybe_get_rope(layer)


class ShiftingKVCache(RotatingKVCache):
    def __init__(self, rope: nn.Module, max_size=None, keep=0, step=256):
        self.rope = rope
        self.reuse_offset = 0
        self.reuse_queue = []
        super().__init__(self, max_size, keep, step)
    
    def is_trimmable(self) -> bool:
        return True
    
    def _trim(self, trim_size, v, append=None):
        to_cat = []
        shift_by = -trim_size
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], self.rope(v[..., trim_size + self.keep :, :], shift_by)]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)
    
    def _temporal_order(self, v) -> mx.array:
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            shift_by = self.keep - self.idx
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    # TODO(christian-lms): verify that i work
                    # TODO(christian-lms): can you do this in 1 call to self.rope?
                    # N.B. this implicitly assumes the generation has not gone over twice
                    # the size of the rotating section of the cache, in which case the
                    # rotating section would be off by a multiple of (max_kv_size - keep)
                    # depending on how many times it rolled over. I feel like it's pretty
                    # safe to assume that this is a rare case
                    self.rope(v[..., self._idx :, :], shift_by),
                    self.rope(v[..., self.keep : self._idx, :], shift_by),
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]
    
    def reuse_section(self, write_start_idx: int, reuse_start_idx: int, reuse_length: int) -> None:
        # offset indices to account for the fact that we move cache elements around
        write_start_idx -= self.reuse_offset
        reuse_start_idx -= self.reuse_offset
        
        # update position offsets for future reuse sections
        shift_by = write_start_idx - reuse_start_idx
        self.reuse_offset += shift_by

        # queue for reuse: everything is done in one pass at the end in do_reuse
        self.reuse_queue.append((write_start_idx, reuse_start_idx, reuse_length))
    
    def do_reuse(self) -> None:
        last_i: int = len(self.reuse_queue) - 1
        for i, (write_start_idx, reuse_start_idx, reuse_length) in enumerate(self.reuse_queue):
            shift_by: int = write_start_idx - reuse_start_idx  # < 0
            reuse_end_idx: int = reuse_start_idx + reuse_length

            keys_to_shift = self.keys[..., reuse_start_idx : reuse_end_idx, :]
            values_to_shift = self.values[..., reuse_start_idx : reuse_end_idx, :]

            # perform rope shift
            # N.B. we can also go back to the MLX-native "don't rope shift" method
            # by 
            shifted_keys = self.rope(keys_to_shift, shift_by)
            shifted_values = self.rope(values_to_shift, shift_by)
            
            # restructure cache with mx.concat
            # TODO(christian-lms): maybe it would be better to use inplace ops.
            # look into the mlx docs if that's even a thing
            keycat = [
                self.keys[..., : write_start_idx, :],
                shifted_keys
            ]
            valcat = [
                self.values[..., : write_start_idx, :],
                shifted_values
            ]
            
            # by not re-appending the end at the last one, we truncate the leftovers
            if i != last_i:
                keycat.append(self.keys[..., reuse_end_idx : , :])
                valcat.append(self.values[..., reuse_end_idx : , :])

            self.keys = mx.concat(keycat, axis=2)
            self.values = mx.concat(valcat, axis=2)
            
            self.offset -= shift_by
        self.reuse_offset = 0
        self.reuse_queue = []
        # TODO(christian-lms): dunno if this number is correct/reasonable/whatever
        self._idx = self.keys.shape[2]
    
    def trim(self, n) -> int:
        # TODO(christian-lms): fix me
        n = min(self.offset, n)
        if n == 0:
            return 0
        
        if self.offset >= self.max_size:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            n = n % (self.max_size - self.keep)

        # do trim: put us back into the state before the circular buffer is full
        new_length = self.keys.shape[2] - n
        self.keys = self.keys[..., :new_length, :]
        self.values = self.values[..., :new_length, :]
        
        self.offset -= n
        self._idx = new_length
        return n
    
def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.
    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.
    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``TrimmableRotatingKVCache`` is used
            with a maximum size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()
    num_layers = len(model.layers)
    if max_kv_size is not None:
        cache = []
        for layer in range(num_layers):
            rope = maybe_get_rope(model, layer)
            if rope is None:
                return [KVCache() for _ in range(num_layers)]
            # TODO(christian-lms): change keep on the fly, must be setattr elsewhere
            cache.append(ShiftingKVCache(rope, max_size=max_kv_size, keep=4))
        return cache
    else:
        return [KVCache() for _ in range(num_layers)]


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
        num_tokens_to_trim = self.cache[0].offset - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                log_warn(
                    prefix="CacheWrapper",
                    message=f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: "
                    "Cache is not trimmable. Clearing the cache instead.",
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
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
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
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

            def prompt_progress_callback(x):
                return None
            
        # TODO(christian-lms): truncation logic goes here now

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
        # TODO(christian-lms): ensure that this works as intended when over length
        # TODO(christian-lms): verify rolling window and truncate middle have n_keep as below
        # TODO(christian-lms): this won't work until we pipe in keep from generate
        n_keep = self.cache[0].keep
        # this behavior is common to rolling window (n_keep = 0) and truncate middle
        # (n_keep > 0), and we should never get here with stop at max
        if len(self.tokens) >= n_keep:
            self.tokens = mx.concat([self.tokens[:n_keep], self.tokens[n_keep+1:]])
        self.tokens = mx.concat([self.tokens, mx.array([token])])
