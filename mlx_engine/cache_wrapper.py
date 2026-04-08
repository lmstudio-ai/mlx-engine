import copy
import logging
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)


PROMPT_PROCESSING_CHUNK_SIZE = 2048

logger = logging.getLogger(__name__)


def validate_prefill_step_size(prefill_step_size: Optional[int] = None) -> int:
    if prefill_step_size is None:
        return PROMPT_PROCESSING_CHUNK_SIZE
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size < 1
    ):
        raise ValueError("prefill_step_size must be a positive integer")
    return prefill_step_size


class CacheWrapper:
    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int,
        checkpoint_tail_tokens: int = 4,  # Checkpoint N tokens before end of prompt
    ):
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.chunk_size = chunk_size
        self.checkpoint_tail_tokens = checkpoint_tail_tokens
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

        self._history = self._make_history()
        self._history_key = "session"
        self._live_tokens: Optional[mx.array] = None
        self._live_cache: List[Any] = self._make_cache()

    @property
    def cache(self) -> List[Any]:
        return self._live_cache

    def _make_cache(self) -> List[Any]:
        cache = make_prompt_cache(self.model, self.max_kv_size)
        if self.draft_model is not None:
            cache += make_prompt_cache(self.draft_model)
        return cache

    def _make_history(self) -> LRUPromptCache:
        # Store up to 4 checkpoints. This number can be tuned (or made configurable) if
        # it's too high or low
        history_capacity = 4
        return LRUPromptCache(max_size=history_capacity)

    def _num_tokens_in_cache(self, cache: Optional[List[Any]] = None) -> int | None:
        cache = self._live_cache if cache is None else cache
        for entry in cache:
            if hasattr(entry, "offset"):
                return entry.offset
        return None

    def _store_snapshot(self, tokens: mx.array, cache: List[Any]) -> None:
        if tokens.size == 0:
            return
        self._history.insert_cache(
            self._history_key,
            tokens.tolist(),
            copy.deepcopy(cache),
        )

    def _flush_live_cache(self) -> None:
        if self._live_tokens is None:
            return

        cache_length = self._num_tokens_in_cache()
        if cache_length is None:
            logger.warning(
                "Could not determine the number of tokens in the live cache. Resetting it."
            )
            self._live_tokens = None
            self._live_cache = self._make_cache()
            return
        if cache_length > len(self._live_tokens):
            logger.warning(
                "The live cache is longer than the tracked token history. Resetting it."
            )
            self._live_tokens = None
            self._live_cache = self._make_cache()
            return
        if cache_length <= 0:
            return

        self._store_snapshot(self._live_tokens[:cache_length], self._live_cache)

    def _restore_cache(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[Optional[List[Any]], mx.array]:
        if len(prompt_tokens) == 0:
            return None, prompt_tokens

        cache, rest = self._history.fetch_nearest_cache(
            self._history_key,
            prompt_tokens.tolist(),
        )
        if cache is not None:
            if len(rest) > 0:
                return cache, prompt_tokens[len(prompt_tokens) - len(rest) :]

            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                return cache, prompt_tokens[-1:]

        if len(prompt_tokens) <= 1:
            return None, prompt_tokens

        # Exact hits need one token outside the cache to seed decode. If the
        # exact-hit cache cannot be trimmed, retry with one less prompt token
        # so a stored checkpoint can win.
        truncated_prompt = prompt_tokens[:-1]
        cache, rest = self._history.fetch_nearest_cache(
            self._history_key,
            truncated_prompt.tolist(),
        )
        if cache is None:
            return None, prompt_tokens

        prefix_length = len(truncated_prompt) - len(rest)
        return cache, prompt_tokens[prefix_length:]

    def _prefill_cache(
        self,
        model: nn.Module,
        cache: List[Any],
        cache_start: int,
        tokens: mx.array,
        reporter: PromptProgressReporter,
        is_draft: bool,
        checkpoint_prefix_len: Optional[int] = None,
    ) -> None:
        remaining_tokens = tokens
        num_processed = 0
        stored_checkpoint = False

        while remaining_tokens.size > 0:
            current_chunk_size = min(self.chunk_size, remaining_tokens.size)
            current_cache_size = self._num_tokens_in_cache(cache)
            if (
                checkpoint_prefix_len is not None
                and current_cache_size is not None
                and current_cache_size < checkpoint_prefix_len
                and current_cache_size + current_chunk_size > checkpoint_prefix_len
            ):
                current_chunk_size = checkpoint_prefix_len - current_cache_size

            current_chunk = remaining_tokens[:current_chunk_size]
            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self.kv_cache_qtn_params)
            self._live_cache[cache_start : cache_start + len(cache)] = cache
            mx.eval([entry.state for entry in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size
            mx.clear_cache()

            current_cache_size = self._num_tokens_in_cache(cache)
            if (
                checkpoint_prefix_len is not None
                and not stored_checkpoint
                and current_cache_size == checkpoint_prefix_len
            ):
                self._store_snapshot(
                    self._live_tokens[:checkpoint_prefix_len],
                    self._live_cache,
                )
                stored_checkpoint = True

            if not reporter.update(is_draft, num_processed):
                logger.info("Prompt processing was cancelled by the user.")
                live_cache_size = self._num_tokens_in_cache()
                if live_cache_size is None:
                    self._live_tokens = None
                    self._live_cache = self._make_cache()
                else:
                    self._live_tokens = self._live_tokens[:live_cache_size]
                raise StopPromptProcessing

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
    ) -> mx.array:
        total_prompt_tokens = len(prompt_tokens)

        self._flush_live_cache()

        restored_cache, uncached_tokens = self._restore_cache(prompt_tokens)
        self._live_cache = (
            restored_cache if restored_cache is not None else self._make_cache()
        )
        self._live_tokens = prompt_tokens

        cached_tokens = total_prompt_tokens - len(uncached_tokens)

        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Leave one token outside the cache to seed decode.
        prefill_tokens = uncached_tokens[:-1]
        checkpoint_prefix_len = None
        # Only checkpoint the main-model path; quantized caches skip checkpointing.
        if self.draft_model is None and self.kv_cache_qtn_params["kv_bits"] is None:
            checkpoint_prefix_len = total_prompt_tokens - self.checkpoint_tail_tokens
            # Skip checkpoints that are already cached or would be empty.
            if checkpoint_prefix_len <= cached_tokens:
                checkpoint_prefix_len = None
            if checkpoint_prefix_len is not None and checkpoint_prefix_len <= 0:
                checkpoint_prefix_len = None

        with mx.stream(generation_stream):
            if self.draft_model is not None:
                draft_cache = self._live_cache[len(self.model.layers) :]
                self._prefill_cache(
                    model=self.draft_model,
                    cache=draft_cache,
                    cache_start=len(self.model.layers),
                    tokens=prefill_tokens,
                    reporter=reporter,
                    is_draft=True,
                    checkpoint_prefix_len=None,
                )

            main_cache = self._live_cache[: len(self.model.layers)]
            self._prefill_cache(
                model=self.model,
                cache=main_cache,
                cache_start=0,
                tokens=prefill_tokens,
                reporter=reporter,
                is_draft=False,
                checkpoint_prefix_len=checkpoint_prefix_len,
            )

        reporter.finish(is_draft=False)
        return uncached_tokens[-1:]

    def record_generated_token(self, token: int) -> None:
        if self._live_tokens is None:
            self._live_tokens = mx.array([token])
            return
        self._live_tokens = mx.concat([self._live_tokens, mx.array([token])])

    def set_draft_model(self, draft_model: nn.Module) -> None:
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self.draft_model is draft_model:
            return
        if self.max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self.max_kv_size = None

        self._history = self._make_history()
        self.draft_model = draft_model
        self._live_tokens = None
        self._live_cache = self._make_cache()

    def unset_draft_model(self) -> None:
        if self.draft_model is None:
            return
        main_cache = self._live_cache[: len(self.model.layers)]
        self._history = self._make_history()
        self.draft_model = None
        if len(main_cache) == len(self.model.layers):
            self._live_cache = main_cache
            return
        self._live_tokens = None
        self._live_cache = self._make_cache()
