import mlx.core as mx
import mlx.nn as nn

from mlx_engine.prompt_cache_session import (
    PreparedPrompt,
    PromptCacheSession,
)
from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
)


class CacheWrapper:
    """
    Backwards-compatible adapter around PromptCacheSession.

    The real implementation lives in `prompt_cache_session.py`.
    """

    def __init__(
        self,
        model: nn.Module,
        max_kv_size: int | None,
        *,
        verbose: bool = False,
        kv_bits: int | None = None,
        kv_group_size: int | None = None,
        quantized_kv_start: int | None = None,
        chunk_size: int,
    ):
        self._session = PromptCacheSession(
            model=model,
            max_kv_size=max_kv_size,
            verbose=verbose,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            chunk_size=chunk_size,
        )

    @property
    def cache(self):
        return self._session._live_cache

    @property
    def model(self):
        return self._session.model

    @property
    def draft_model(self):
        return self._session.draft_model

    def set_draft_model(self, draft_model: nn.Module):
        self._session.set_draft_model(draft_model)

    def unset_draft_model(self):
        self._session.unset_draft_model()

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        prepared: PreparedPrompt = self._session.prepare(
            prompt_tokens,
            reporter,
            num_tokens_to_exclude=num_tokens_to_exclude,
        )
        return prepared.uncached_tokens

    def record_generated_token(self, token: int):
        self._session.record_generated_token(token)
