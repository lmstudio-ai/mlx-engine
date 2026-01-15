from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter


@dataclass
class PromptProgressBeginEvent:
    cached_tokens: int
    total_prompt_tokens: int
    prefill_tokens_processed: int


@dataclass
class PromptProgressEvent:
    prefill_tokens_processed: int
    is_final: bool = False


@dataclass
class ProgressContext:
    cached_tokens: int
    total_prompt_tokens: int
    last_prefill_tokens_processed: int = 0


PromptProgressCallbackEvent = PromptProgressBeginEvent | PromptProgressEvent


class PromptProgressCallback(Protocol):
    def __call__(self, event: PromptProgressCallbackEvent, is_draft: bool) -> bool: ...


PercentPromptProgressCallback = Callable[[float], None]


class PromptProgressCallbackReporter(PromptProgressReporter):
    """
    Reporter that emits event-based progress via a callback, with optional percent reporting.

    Draft events only invoke the progress callback; percent reporting is suppressed.
    Only the progress callback return value is respected for should-continue.
    """

    def __init__(
        self,
        progress_callback: PromptProgressCallback,
        *,
        percent_callback: Optional[PercentPromptProgressCallback] = None,
    ):
        self._progress_callback = progress_callback
        self._percent_callback = percent_callback
        self._context: Optional[ProgressContext] = None

    def _emit_percent(self, prefill_tokens_processed: int) -> None:
        if self._percent_callback is None or self._context is None:
            return
        tokens_to_prefill = (
            self._context.total_prompt_tokens - self._context.cached_tokens
        )
        if tokens_to_prefill <= 0:
            self._percent_callback(100.0)
        else:
            percent = (prefill_tokens_processed / tokens_to_prefill) * 100.0
            self._percent_callback(min(100.0, max(0.0, percent)))

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        event = PromptProgressBeginEvent(
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=prefill_tokens_processed,
        )
        should_continue = self._progress_callback(event, is_draft)
        if not is_draft:
            self._context = ProgressContext(
                cached_tokens, total_prompt_tokens, prefill_tokens_processed
            )
            self._emit_percent(prefill_tokens_processed)
        return should_continue

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        event = PromptProgressEvent(prefill_tokens_processed=prefill_tokens_processed)
        should_continue = self._progress_callback(event, is_draft)
        if not is_draft:
            if self._context is not None:
                self._context.last_prefill_tokens_processed = prefill_tokens_processed
            self._emit_percent(prefill_tokens_processed)
        return should_continue

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        if prefill_tokens_processed is None and self._context is not None:
            prefill_tokens_processed = self._context.last_prefill_tokens_processed
        event = PromptProgressEvent(
            prefill_tokens_processed=prefill_tokens_processed or 0,
            is_final=True,
        )
        should_continue = self._progress_callback(event, is_draft)
        if not is_draft and prefill_tokens_processed is not None:
            self._emit_percent(prefill_tokens_processed)
        return should_continue
