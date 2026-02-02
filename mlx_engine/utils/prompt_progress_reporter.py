from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StopPromptProcessing(Exception):
    """
    Exception to signal that the user aborted generation during prompt processing.
    """


class PromptProgressReporter(ABC):
    """
    Reporter for receiving prompt processing progress updates.

    Subclass this to implement custom progress handling.
    """

    @abstractmethod
    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        """
        Called when prompt processing starts.

        Args:
            is_draft: True if this is for the draft model, False for main model.
            cached_tokens: Number of tokens already in the KV cache.
            total_prompt_tokens: Total number of tokens in the prompt.
            prefill_tokens_processed: Number of tokens processed so far (usually 0 at begin).

        Returns:
            True to continue processing, False to cancel.
        """
        ...

    @abstractmethod
    def update(
        self,
        is_draft: bool,
        prefill_tokens_processed: int,
    ) -> bool:
        """
        Called during processing with progress updates.

        Args:
            is_draft: True if this is for the draft model, False for main model.
            prefill_tokens_processed: Number of tokens processed so far.

        Returns:
            True to continue processing, False to cancel.
        """
        ...

    @abstractmethod
    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        """
        Called when prompt processing completes.

        Args:
            is_draft: True if this is for the draft model, False for main model.
            prefill_tokens_processed: Total number of tokens processed, or None if not available.

        Returns:
            True to continue processing, False to cancel.
        """
        ...


class DefaultPromptProgressReporter(PromptProgressReporter):
    """A no-op reporter that always continues processing."""

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        return True

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        return True


class LoggerReporter(PromptProgressReporter):
    """A reporter that logs all events it receives."""

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        logger.info(
            f"begin: is_draft={is_draft}, cached_tokens={cached_tokens}, "
            f"total_prompt_tokens={total_prompt_tokens}, "
            f"prefill_tokens_processed={prefill_tokens_processed}"
        )
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        logger.info(
            f"update: is_draft={is_draft}, "
            f"prefill_tokens_processed={prefill_tokens_processed}"
        )
        return True

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        logger.info(
            f"finish: is_draft={is_draft}, "
            f"prefill_tokens_processed={prefill_tokens_processed}"
        )
        return True


class ForwardingReporter(PromptProgressReporter):
    """
    Wrapper that raises StopPromptProcessing when the inner reporter signals cancellation.
    """

    def __init__(
        self, inner: PromptProgressReporter, *, raise_error_when_stopped: bool
    ):
        self._inner = inner
        self._raise_error_when_stopped = raise_error_when_stopped

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        should_continue = self._inner.begin(
            is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed
        )
        if not should_continue:
            logger.info("Prompt processing was cancelled by the user.")
            if self._raise_error_when_stopped:
                raise StopPromptProcessing
        return should_continue

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        should_continue = self._inner.update(is_draft, prefill_tokens_processed)
        if not should_continue:
            logger.info("Prompt processing was cancelled by the user.")
            if self._raise_error_when_stopped:
                raise StopPromptProcessing
        return should_continue

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        should_continue = self._inner.finish(is_draft, prefill_tokens_processed)
        if not should_continue:
            logger.info("Prompt processing was cancelled by the user.")
            if self._raise_error_when_stopped:
                raise StopPromptProcessing
        return should_continue


class MlxLmReporterAdapter:
    """
    Adapts a PromptProgressReporter to the mlx-lm callback signature.

    Converts (processed_tokens, total_tokens) -> None to reporter method calls.
    Automatically calls finish() when processed_tokens >= total_tokens.

    Wraps the reporter with ThrowToStopReporter to convert cancellation (returning False)
    into a StopPromptProcessing exception, since mlx-lm's callback signature doesn't
    support return-value-based cancellation.
    """

    def __init__(self, reporter: PromptProgressReporter, emit_begin: bool = False):
        self._reporter = ForwardingReporter(reporter, raise_error_when_stopped=True)
        self._emit_begin = emit_begin
        self._first_call = True
        self._finished = False

    def __call__(self, processed_tokens: int, total_tokens: int) -> None:
        if self._finished:
            return

        if self._first_call:
            self._first_call = False
            if self._emit_begin:
                self._reporter.begin(
                    is_draft=False,
                    cached_tokens=0,
                    total_prompt_tokens=total_tokens,
                    prefill_tokens_processed=processed_tokens,
                )
                # Don't also emit update on the same call as begin
                return

        if processed_tokens >= total_tokens:
            self._finished = True
            self._reporter.finish(
                is_draft=False, prefill_tokens_processed=processed_tokens
            )
        else:
            self._reporter.update(
                is_draft=False, prefill_tokens_processed=processed_tokens
            )


class BatchedMlxLmReporterAdapter:
    """
    Adapts a PromptProgressReporter to the BatchedModelKit.generate callback.

    Converts (processed_tokens, total_tokens) -> None to reporter method calls.
    Automatically calls finish() when processed_tokens - 1 >= total_tokens.
    We need the off-by-one since mlx-lm prefills every token except for the last one,
    since that token is needed to start the auto-regressive decoding

    Unlike MlxLmReporterAdapter, do not throw when we receive a stop request. This should be fixed
    when batched mlx-lm supports prompt processing interruption
    """

    def __init__(self, reporter: PromptProgressReporter, emit_begin: bool = False):
        # TODO: need a way to interrupt prompt processing
        self._reporter = ForwardingReporter(reporter, raise_error_when_stopped=False)

        self._emit_begin = emit_begin
        self._first_call = True
        self._finished = False

    def __call__(self, processed_tokens: int, total_tokens: int) -> None:
        if self._finished:
            return

        # mlx-lm tells us how many total prompt tokens there are. It leaves one unprocessed to seed the decode. Make that adjustment here
        total_tokens = max(0, total_tokens - 1)

        if self._first_call:
            self._first_call = False
            if self._emit_begin:
                self._reporter.begin(
                    is_draft=False,
                    cached_tokens=0,  # This is likely wrong
                    total_prompt_tokens=total_tokens,
                    prefill_tokens_processed=processed_tokens,
                )

        if processed_tokens >= total_tokens:
            self._finished = True
            self._reporter.finish(
                is_draft=False, prefill_tokens_processed=processed_tokens
            )
        else:
            self._reporter.update(
                is_draft=False, prefill_tokens_processed=processed_tokens
            )
