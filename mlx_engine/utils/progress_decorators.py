from typing import Callable, Union
from mlx_engine.cache_wrapper import StopPromptProcessing
from mlx_engine.utils.prompt_progress_events import (
    PromptProgressBeginEvent,
    PromptProgressEvent,
    V2ProgressCallback,
)
import logging

logger = logging.getLogger(__name__)


def default_callback(event: Union[PromptProgressBeginEvent, PromptProgressEvent], is_draft: bool) -> bool:
    """
    A no-op callback that always returns True to continue processing.

    Use this as a fallback when a callback is required but none was provided.
    """
    return True


def ratchet(
    callback: V2ProgressCallback,
) -> V2ProgressCallback:
    """
    Wraps a V2 progress callback to ensure progress values are monotonically increasing.

    This wrapper prevents progress from appearing to move backwards by using a ratchet
    mechanism. If a lower token count is reported than previously seen, the callback
    returns True (continue) without calling the original callback.

    Args:
        callback: A V2 callback that accepts PromptProgressBeginEvent or PromptProgressEvent
                 and returns True to continue or False to stop.

    Returns:
        A wrapped callback that ensures monotonic progress reporting.
    """
    max_tokens_processed = -1

    def inner_callback(event: Union[PromptProgressBeginEvent, PromptProgressEvent], is_draft: bool) -> bool:
        nonlocal max_tokens_processed
        
        # Always pass through BeginEvent (it resets the ratchet)
        if isinstance(event, PromptProgressBeginEvent):
            max_tokens_processed = event.prefill_tokens_processed
            return callback(event, is_draft)
        
        # For ProgressEvent, check if we've regressed
        if isinstance(event, PromptProgressEvent):
            if event.prefill_tokens_processed <= max_tokens_processed:
                return True  # Skip callback, continue processing
            max_tokens_processed = event.prefill_tokens_processed
            return callback(event, is_draft)
        
        # Unknown event type, pass through
        return callback(event, is_draft)

    return inner_callback


def throw_to_stop(
    callback: V2ProgressCallback,
) -> V2ProgressCallback:
    """
    Wraps a V2 progress callback to raise an exception when stopping is requested.

    Instead of returning False to indicate stopping, this wrapper raises a
    StopPromptProcessing exception when the original callback returns False.
    This allows for immediate termination of the generation process.

    Args:
        callback: A V2 callback that accepts PromptProgressBeginEvent or PromptProgressEvent
                 and returns True to continue or False to stop.

    Returns:
        A wrapped callback that raises StopPromptProcessing when stopping
        is requested.

    Raises:
        StopPromptProcessing: When the original callback returns False.
    """
    def inner_callback(event: Union[PromptProgressBeginEvent, PromptProgressEvent], is_draft: bool) -> bool:
        should_continue = callback(event, is_draft)
        if not should_continue:
            logger.info("Prompt processing was cancelled by the user.")
            raise StopPromptProcessing
        return should_continue

    return inner_callback


def mlx_lm_converter(
    callback: V2ProgressCallback,
    *,
    emit_begin_event: bool = False,
) -> Callable[[int, int], None]:
    """
    Adapts a V2 progress callback into a token count based callback for mlx-lm.

    This decorator converts the (processed_tokens, total_tokens) callback signature
    used by mlx-lm's stream_generate into V2 PromptProgressEvent objects.

    Args:
        callback: A V2 callback that accepts PromptProgressBeginEvent or PromptProgressEvent
                 and returns True to continue or False to stop.
        emit_begin_event: If True, emits a BeginEvent on the first callback invocation.
                         Used for vision processing where mlx-lm calculates the correct
                         total token count (including expanded image tokens).

    Returns:
        A token-based callback (processed_tokens, total_tokens) -> None,
        as is expected by mlx-lm's stream_generate.
    """
    first_call = True
    
    def inner_callback(processed_tokens: int, total_tokens: int) -> None:
        nonlocal first_call
        
        # Emit BeginEvent on first call if requested (for vision processing)
        if first_call and emit_begin_event:
            first_call = False
            begin_event = PromptProgressBeginEvent(
                cached_tokens=0,
                total_prompt_tokens=total_tokens,
                prefill_tokens_processed=0
            )
            callback(begin_event, is_draft=False)
        
        else:
            # Emit progress event with current token count
            event = PromptProgressEvent(prefill_tokens_processed=processed_tokens)
            callback(event, is_draft=False)

    return inner_callback
