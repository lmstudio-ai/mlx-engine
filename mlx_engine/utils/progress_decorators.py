import logging
import time
from typing import Callable, Optional, Union

from mlx_engine.cache_wrapper import StopPromptProcessing

logger = logging.getLogger(__name__)


def backward_compatible(
    callback: Optional[Callable[[float], Union[bool, None]]],
) -> Optional[Callable[[float], bool]]:
    """
    Wraps a progress callback to ensure backward compatibility with old and new callback patterns.

    This wrapper handles both old-style callbacks that return None (or no explicit return)
    and new-style callbacks that return bool. It ensures all callbacks return a boolean
    value as expected by the current implementation.

    Args:
        callback: A callback that accepts progress (0.0–100.0) and may return
                  True/False (new style) or None/no return (old style). May be None.

    Returns:
        A wrapped callback that always returns a boolean value.
        If callback is None, returns None.
    """
    if callback is None:
        return None

    def inner_callback(percentage: float) -> bool:
        try:
            result = callback(percentage)
            # Handle old-style callbacks that return None or no explicit return
            if result is None:
                return True  # Default to continue for old-style callbacks
            return bool(result)  # Ensure boolean return for new-style callbacks
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}. Continuing processing.")
            return True  # Continue processing if callback fails

    return inner_callback


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


def synthetic_progress(
    callback: Optional[Callable[[float], bool]],
    total_tokens: int,
    tick_count: int = 5,
) -> Optional[Callable[[float], bool]]:
    """
    Creates a synthetic progress callback for unbounded prefill.

    This wrapper generates synthetic progress ticks during unbounded prefill
    to provide user feedback while the actual prefill happens in one pass.

    Args:
        callback: A callback that accepts progress (0.0–100.0) and returns
                  True to continue or False to stop. May be None.
        total_tokens: Total number of tokens being processed
        tick_count: Number of synthetic progress ticks to generate

    Returns:
        A callback that when called will emit synthetic progress.
        If callback is None, returns None.
    """
    if callback is None:
        return None

    # Generate synthetic progress ticks
    if tick_count < 2:
        ticks = [0.0, 100.0]
    else:
        ticks = []
        for i in range(tick_count):
            if i == 0:
                ticks.append(0.0)
            elif i == tick_count - 1:
                ticks.append(100.0)
            else:
                # Use quadratic progression for more realistic progress
                progress = (i / (tick_count - 1)) ** 2 * 100.0
                ticks.append(progress)

    # Calculate timing for synthetic progress
    start_time = time.time()
    total_duration = 2.0  # Assume 2 seconds for unbounded prefill
    current_tick = 0

    def inner_callback(progress: float) -> bool:
        nonlocal current_tick

        # Ignore the actual progress and emit synthetic progress
        elapsed = time.time() - start_time
        if elapsed >= total_duration:
            synthetic_progress = 100.0
        else:
            # Find the appropriate tick based on elapsed time
            progress_ratio = elapsed / total_duration
            tick_index = int(progress_ratio * (len(ticks) - 1))
            synthetic_progress = ticks[min(tick_index, len(ticks) - 1)]

        current_tick = min(current_tick + 1, len(ticks) - 1)
        return callback(synthetic_progress)

    return inner_callback
