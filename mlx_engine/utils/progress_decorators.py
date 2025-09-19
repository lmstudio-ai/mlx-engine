from typing import Optional, Callable
from mlx_engine.cache_wrapper import StopPromptProcessing
import logging

logger = logging.getLogger(__name__)


def ratchet(
    callback: Optional[Callable[[float], bool]],
) -> Optional[Callable[[float], bool]]:
    """
    Wraps a progress callback to ensure progress values are monotonically increasing.

    This wrapper prevents progress from appearing to move backwards by using a ratchet
    mechanism. If a lower percentage is reported than previously seen, the callback
    returns True (continue) without calling the original callback.

    Args:
        callback: A callback that accepts progress (0.0–100.0) and returns
                 True to continue or False to stop. May be None.

    Returns:
        A wrapped callback that ensures monotonic progress reporting.
        If callback is None, returns None.
    """
    if callback is None:
        return None

    ratchet = float("-inf")

    def inner_callback(percentage: float) -> bool:
        nonlocal ratchet
        if percentage <= ratchet:
            return True
        ratchet = percentage
        return callback(percentage)

    return inner_callback


def throw_to_stop(
    callback: Optional[Callable[[float], bool]],
) -> Optional[Callable[[float], bool]]:
    """
    Wraps a progress callback to raise an exception when stopping is requested.

    Instead of returning False to indicate stopping, this wrapper raises a
    StopPromptProcessing exception when the original callback returns False.
    This allows for immediate termination of the generation process.

    Args:
        callback: A callback that accepts progress (0.0–100.0) and returns
                 True to continue or False to stop. May be None.

    Returns:
        A wrapped callback that raises StopPromptProcessing when stopping
        is requested. If callback is None, returns None.

    Raises:
        StopPromptProcessing: When the original callback returns False.
    """
    if callback is None:
        return None

    def inner_callback(percentage: float) -> bool:
        should_continue = callback(percentage)
        if not should_continue:
            logger.info("Prompt processing was cancelled by the user.")
            raise StopPromptProcessing
        return should_continue

    return inner_callback


def token_count(
    callback: Optional[Callable[[float], bool]],
) -> Optional[Callable[[int, int], None]]:
    """
    Adapts a float percentage based progress callback into a token count based one.

    Args:
        outer_callback: A callback that accepts progress (0.0–100.0) and returns
                        True to continue or False to stop. May be None.

    Returns:
        A token-based callback (processed_tokens, total_tokens) -> None,
        as is expected by mlx-lm's stream_generate.
        If outer_callback is None, returns None.
    """
    if callback is None:
        return None

    def inner_callback(processed_tokens: int, total_tokens: int) -> None:
        if total_tokens <= 0:
            progress = 0.0
        else:
            progress = 100 * processed_tokens / total_tokens
        callback(progress)

    return inner_callback
