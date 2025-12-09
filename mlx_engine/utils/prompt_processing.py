import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

# Conditional imports for mlx dependencies
MLX_AVAILABLE = False
try:
    import mlx.core as mx  # type: ignore
    from mlx import nn  # type: ignore

    MLX_AVAILABLE = True
except ImportError:
    # Create stub classes for type hints when mlx is not available
    class nn:
        class Module:
            pass

    class mx_module:
        @staticmethod
        def array(data):
            return data

    class mx:
        class array:
            pass


# Initialize logger
logger = logging.getLogger(__name__)

# Import structured logger for decision logging
get_structured_logger = None
try:
    from mlx_engine.utils.logger import get_structured_logger  # type: ignore

    STRUCTURED_LOGGER_AVAILABLE = True
except ImportError:
    STRUCTURED_LOGGER_AVAILABLE = False


@dataclass
class PerformanceProfileCompat:
    """PerformanceProfile compatible with test expectations."""

    name: str
    prefill_mode: str
    unbounded_allowed: bool
    cache_slots: int
    chunk_size_min: int
    chunk_size_max: int
    kv_bytes_per_token_estimate: int = 2048
    max_prefill_tokens_per_pass: int = 8192


@dataclass
class PrefillPlan:
    """Plan for prefill strategy execution."""

    mode: str
    chunk_size: Optional[int]
    reason: Optional[str]
    total_chunks: int
    synthetic_progress_ticks: int


def calculate_memory_pressure(
    required_mem_bytes: int, available_mem_bytes: int
) -> tuple[float, str]:
    """
    Calculate memory pressure ratio and categorize pressure level.

    Args:
        required_mem_bytes: Memory required for the operation
        available_mem_bytes: Available memory in bytes

    Returns:
        Tuple of (pressure_ratio, pressure_category)
        pressure_ratio: required/available (0.0-1.0+)
        pressure_category: "low", "medium", "high", "critical"
    """
    if available_mem_bytes <= 0:
        return float("inf"), "critical"

    pressure_ratio = required_mem_bytes / available_mem_bytes

    if pressure_ratio < 0.5:
        return pressure_ratio, "low"
    elif pressure_ratio < 0.7:
        return pressure_ratio, "medium"
    elif pressure_ratio < 0.9:
        return pressure_ratio, "high"
    else:
        return pressure_ratio, "critical"


def calculate_adaptive_chunk_size(
    prompt_tokens: int,
    profile: PerformanceProfileCompat,
    kv_bytes_per_token: int,
    available_mem_bytes: int,
) -> tuple[int, str]:
    """
    Calculate adaptive chunk size based on memory constraints and prompt characteristics.

    Args:
        prompt_tokens: Number of prompt tokens
        profile: Performance profile
        kv_bytes_per_token: KV cache bytes per token
        available_mem_bytes: Available memory in bytes

    Returns:
        Tuple of (chunk_size, decision_reason)
    """
    # Base memory calculation with 80% headroom
    base_mem_available = available_mem_bytes * 0.8
    base_chunk_size = int(base_mem_available / kv_bytes_per_token)

    # Calculate memory pressure
    required_mem = prompt_tokens * kv_bytes_per_token
    pressure_ratio, pressure_category = calculate_memory_pressure(
        required_mem, available_mem_bytes
    )

    # Prompt length scaling
    if prompt_tokens < 1000:
        prompt_category = "short"
        prompt_scale_factor = 1.2  # Allow larger chunks for short prompts
    elif prompt_tokens < 10000:
        prompt_category = "medium"
        prompt_scale_factor = 1.0  # Standard scaling
    else:
        prompt_category = "long"
        prompt_scale_factor = 0.8  # Smaller chunks for long prompts

    # Memory pressure adaptation
    if pressure_category == "low":
        headroom_factor = 1.0  # Full headroom available
        pressure_desc = "low pressure"
    elif pressure_category == "medium":
        headroom_factor = 0.85  # 15% reduction
        pressure_desc = "medium pressure"
    elif pressure_category == "high":
        headroom_factor = 0.7  # 30% reduction
        pressure_desc = "high pressure"
    else:  # critical
        headroom_factor = 0.5  # 50% reduction
        pressure_desc = "critical pressure"

    # Cache slot multiplier effect
    cache_multiplier = 1.0
    if profile.cache_slots > 1:
        # More cache slots reduce effective chunk size
        cache_multiplier = 1.0 / (1.0 + (profile.cache_slots - 1) * 0.2)
        cache_desc = f"cache slots ({profile.cache_slots}) reduce size"
    else:
        cache_desc = "single cache slot"

    # Calculate adaptive chunk size
    adaptive_chunk_size = int(
        base_chunk_size * prompt_scale_factor * headroom_factor * cache_multiplier
    )

    # Build decision reason
    decision_parts = [
        f"base {base_chunk_size}",
        f"prompt {prompt_category} ({prompt_scale_factor:.1f}x)",
        f"{pressure_desc} ({headroom_factor:.1f}x)",
        cache_desc,
    ]
    decision_reason = "Adaptive chunk: " + ", ".join(decision_parts)

    logger.debug(
        f"Adaptive chunk calculation: tokens={prompt_tokens}, "
        f"pressure={pressure_ratio:.2f} ({pressure_category}), "
        f"base_chunk={base_chunk_size}, adaptive_chunk={adaptive_chunk_size}"
    )

    return adaptive_chunk_size, decision_reason


def enforce_profile_bounds(
    chunk_size: int, profile: PerformanceProfileCompat, context: str
) -> tuple[int, str]:
    """
    Enforce profile min/max bounds with detailed logging.

    Args:
        chunk_size: Calculated chunk size
        profile: Performance profile
        context: Context description for logging

    Returns:
        Tuple of (clamped_chunk_size, bound_action)
    """
    original_size = chunk_size
    bound_actions = []

    # Enforce minimum bound
    if chunk_size < profile.chunk_size_min:
        chunk_size = profile.chunk_size_min
        bound_actions.append(f"raised to min {profile.chunk_size_min}")

    # Enforce maximum bound
    if chunk_size > profile.chunk_size_max:
        chunk_size = profile.chunk_size_max
        bound_actions.append(f"lowered to max {profile.chunk_size_max}")

    if bound_actions:
        bound_action = f"Bounds enforced: {', '.join(bound_actions)}"
        logger.info(
            f"{context}: chunk size {original_size} -> {chunk_size} "
            f"({', '.join(bound_actions)})"
        )
    else:
        bound_action = "No bounds enforcement needed"
        logger.debug(f"{context}: chunk size {chunk_size} within bounds")

    return chunk_size, bound_action


def validate_chunk_safety(
    chunk_size: int,
    profile: PerformanceProfileCompat,
    kv_bytes_per_token: int,
    available_mem_bytes: int,
) -> tuple[bool, Optional[str]]:
    """
    Validate chunk size safety with 30% safety margin.

    Args:
        chunk_size: Proposed chunk size
        profile: Performance profile
        kv_bytes_per_token: KV cache bytes per token
        available_mem_bytes: Available memory in bytes

    Returns:
        Tuple of (is_safe, safety_reason)
    """
    # Calculate memory needed for chunk with safety margin
    chunk_mem_needed = chunk_size * kv_bytes_per_token
    safety_margin = 1.3  # 30% safety margin
    safe_mem_limit = available_mem_bytes / safety_margin

    # Account for cache slots
    cache_multiplier = profile.cache_slots if profile.cache_slots > 1 else 1
    total_chunk_mem = chunk_mem_needed * cache_multiplier

    is_safe = total_chunk_mem <= safe_mem_limit

    if not is_safe:
        safety_reason = (
            f"Chunk {chunk_size} unsafe: needs {total_chunk_mem / (1024**2):.0f}MB "
            f"with safety margin, limit {safe_mem_limit / (1024**2):.0f}MB"
        )
        logger.warning(f"Chunk safety validation failed: {safety_reason}")
    else:
        safety_reason = (
            f"Chunk {chunk_size} safe: uses {total_chunk_mem / (1024**2):.0f}MB"
        )
        logger.debug(f"Chunk safety validation passed: {safety_reason}")

    return is_safe, safety_reason


def log_chunk_decision(
    prompt_tokens: int,
    adaptive_chunk: int,
    final_chunk: int,
    adaptive_reason: str,
    bound_action: str,
    safety_result: tuple[bool, Optional[str]],
    fallback_used: bool = False,
) -> None:
    """
    Log detailed chunk decision path.

    Args:
        prompt_tokens: Number of prompt tokens
        adaptive_chunk: Initial adaptive chunk size
        final_chunk: Final chunk size after bounds and safety
        adaptive_reason: Reason for adaptive calculation
        bound_action: Bounds enforcement action
        safety_result: Tuple of (is_safe, safety_reason)
        fallback_used: Whether fallback to min was used
    """
    is_safe, safety_reason = safety_result

    logger.info(
        f"Chunk decision path: prompt={prompt_tokens} tokens, "
        f"adaptive={adaptive_chunk}, final={final_chunk}"
    )
    logger.info(f"  Adaptive reasoning: {adaptive_reason}")
    logger.info(f"  Bounds action: {bound_action}")
    logger.info(f"  Safety: {safety_reason}")

    if fallback_used:
        logger.warning(
            f"  Fallback: used chunk_size_min={final_chunk} due to safety validation"
        )

    # Calculate total chunks estimate
    total_chunks = (prompt_tokens + final_chunk - 1) // final_chunk
    logger.info(f"  Estimated chunks: {total_chunks}")


def plan_prefill_strategy(
    prompt_tokens: int,
    profile: PerformanceProfileCompat,
    kv_bytes_per_token: int,
    available_mem_bytes: int,
    requested_mode: Optional[str] = None,
    speculative_required: bool = False,
) -> PrefillPlan:
    """
    Plan prefill strategy based on profile and constraints.

    Args:
        prompt_tokens: Number of prompt tokens
        profile: Performance profile
        kv_bytes_per_token: KV cache bytes per token
        available_mem_bytes: Available memory in bytes
        requested_mode: Requested prefill mode
        speculative_required: Whether speculative decoding is required

    Returns:
        PrefillPlan with strategy details

    Raises:
        ValueError: If invalid mode requested or unbounded not allowed
    """
    # Validate requested mode
    if requested_mode and requested_mode not in ["auto", "chunked", "unbounded"]:
        raise ValueError(f"Invalid requested mode: {requested_mode}")

    # Check if unbounded requested but not allowed
    if requested_mode == "unbounded" and not profile.unbounded_allowed:
        raise ValueError("Unbounded mode not allowed for this profile")

    # Determine initial mode
    if speculative_required:
        actual_mode = "chunked"
        reason = "Speculative decoding requires chunked prefill"
    elif requested_mode == "unbounded":
        actual_mode = "unbounded"
        reason = "User requested unbounded mode"
    elif requested_mode == "chunked":
        actual_mode = "chunked"
        reason = "User requested chunked mode"
    elif profile.prefill_mode == "auto":
        # Auto-determine based on constraints
        actual_mode = "unbounded" if profile.unbounded_allowed else "chunked"
        reason = f"Auto mode: selected {actual_mode} based on profile constraints"
    else:
        actual_mode = profile.prefill_mode
        reason = f"Using profile default mode: {actual_mode}"

    # Check headroom for unbounded mode
    if actual_mode == "unbounded":
        # Require 20% headroom for unbounded
        # Account for cache slots in memory calculation
        cache_multiplier = profile.cache_slots if profile.cache_slots > 1 else 1
        required_mem = prompt_tokens * kv_bytes_per_token * cache_multiplier

        # First check if we have enough memory at all
        if required_mem > available_mem_bytes:
            actual_mode = "chunked"
            if cache_multiplier > 1:
                reason = f"Cache slots ({cache_multiplier}) require too much memory for unbounded"
            else:
                reason = f"Insufficient memory for unbounded (required {required_mem / (1024**2):.0f}MB > available {available_mem_bytes / (1024**2):.0f}MB)"
        else:
            headroom_bytes = available_mem_bytes - required_mem
            headroom_ratio = headroom_bytes / required_mem if required_mem > 0 else 0

            # Also check if available memory is too small in absolute terms
            # (e.g., less than or equal to 1GB is considered "tiny" for unbounded)
            if available_mem_bytes <= 1024**3:  # Less than or equal to 1GB
                actual_mode = "chunked"
                reason = f"Insufficient headroom (available {available_mem_bytes / (1024**2):.0f}MB < 1GB) for unbounded"
            elif headroom_ratio < 0.2:
                actual_mode = "chunked"
                reason = (
                    f"Insufficient headroom ({headroom_ratio:.1%} < 20%) for unbounded"
                )
            elif prompt_tokens > profile.max_prefill_tokens_per_pass:
                actual_mode = "chunked"
                reason = f"Chunked mode: prompt tokens ({prompt_tokens}) exceed max unbounded limit ({profile.max_prefill_tokens_per_pass})"
            else:
                # Unbounded mode is suitable
                if reason is None:
                    reason = f"Unbounded mode selected: sufficient memory ({available_mem_bytes / (1024**2):.0f}MB) and headroom ({headroom_ratio:.1%})"

    # Calculate chunk size if needed
    chunk_size: Optional[int] = None
    if actual_mode == "chunked":
        # Use enhanced adaptive chunk sizing
        adaptive_chunk_size, adaptive_reason = calculate_adaptive_chunk_size(
            prompt_tokens, profile, kv_bytes_per_token, available_mem_bytes
        )

        # Enforce profile bounds
        bounded_chunk_size, bound_action = enforce_profile_bounds(
            adaptive_chunk_size, profile, "Chunk sizing"
        )

        # Validate chunk safety
        is_safe, safety_reason = validate_chunk_safety(
            bounded_chunk_size, profile, kv_bytes_per_token, available_mem_bytes
        )

        # Apply fallback strategy if unsafe
        if not is_safe:
            chunk_size = profile.chunk_size_min
            fallback_used = True
            final_reason = f"Fallback to min chunk size: {safety_reason}"
            if reason is None:
                reason = final_reason
        else:
            chunk_size = bounded_chunk_size
            fallback_used = False
            if reason is None:
                reason = f"Adaptive chunk sizing: {adaptive_reason}"

        # Log detailed decision path
        log_chunk_decision(
            prompt_tokens=prompt_tokens,
            adaptive_chunk=adaptive_chunk_size,
            final_chunk=chunk_size,
            adaptive_reason=adaptive_reason,
            bound_action=bound_action,
            safety_result=(is_safe, safety_reason),
            fallback_used=fallback_used,
        )

    # Calculate synthetic progress ticks for unbounded mode
    synthetic_progress_ticks = 0
    if actual_mode == "unbounded":
        # Use 5-10 ticks based on prompt size
        synthetic_progress_ticks = min(10, max(5, prompt_tokens // 1000))

    if actual_mode == "unbounded":
        total_chunks = 1
    else:
        # chunk_size is guaranteed to be an int when actual_mode is "chunked"
        assert chunk_size is not None, (
            "chunk_size must be set when actual_mode is 'chunked'"
        )
        total_chunks = (prompt_tokens + chunk_size - 1) // chunk_size

    # Log decision details if structured logger is available
    if STRUCTURED_LOGGER_AVAILABLE and get_structured_logger is not None:
        try:
            structured_logger = get_structured_logger(__name__)

            # Initialize chunk calculation variables safely
            chunk_calculation = None
            if actual_mode == "chunked":
                # These variables are only defined in chunked mode
                chunk_calculation = {
                    "adaptive_chunk": locals().get("adaptive_chunk_size"),
                    "bounded_chunk": locals().get("bounded_chunk_size"),
                    "safety_validation": locals().get("is_safe"),
                    "fallback_used": locals().get("fallback_used", False),
                }

            decision_details = {
                "prompt_tokens": prompt_tokens,
                "requested_mode": requested_mode,
                "speculative_required": speculative_required,
                "actual_mode": actual_mode,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
                "synthetic_progress_ticks": synthetic_progress_ticks,
                "profile_name": profile.name,
                "profile_prefill_mode": profile.prefill_mode,
                "unbounded_allowed": profile.unbounded_allowed,
                "cache_slots": profile.cache_slots,
                "kv_bytes_per_token": kv_bytes_per_token,
                "available_mem_mb": available_mem_bytes / (1024**2),
                "reason": reason,
                "decision_path": {
                    "mode_selection": actual_mode,
                    "chunk_calculation": chunk_calculation,
                },
            }
            structured_logger.log_decision("prefill_strategy", decision_details)
        except Exception as e:
            logger.debug(f"Failed to log decision details: {e}")

    return PrefillPlan(
        mode=actual_mode,
        chunk_size=chunk_size,
        reason=reason,
        total_chunks=total_chunks,
        synthetic_progress_ticks=synthetic_progress_ticks,
    )


def emit_synthetic_progress(total_tokens: int, tick_count: int) -> list[float]:
    """
    Generate synthetic progress ticks for unbounded prefill.

    Args:
        total_tokens: Total number of tokens to process
        tick_count: Number of progress ticks to generate

    Returns:
        List of progress percentages from 0.0 to 100.0
    """
    if tick_count < 2:
        return [0.0, 100.0]

    # Generate logarithmic-style progress that accelerates
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

    return ticks


# Export for tests
PerformanceProfile = PerformanceProfileCompat


def process_prompt_text_only(
    prompt_tokens: Any,
    cache_wrapper: Any,
    generate_args: dict,
    draft_model: Optional[Any],
    speculative_decoding_toggle: Optional[bool],
    prompt_progress_callback: Optional[Callable[[float], Union[bool, None]]],
) -> Any:
    """
    Process text-only prompts with cache management and speculative decoding support.

    Args:
        prompt_tokens: The prompt tokens to process (mlx.array when available)
        cache_wrapper: Cache wrapper instance for managing KV cache
        generate_args: Generation arguments dict that will be updated with prompt_cache
        draft_model: Optional draft model for speculative decoding
        speculative_decoding_toggle: Optional toggle for speculative decoding
        prompt_progress_callback: Optional callback for progress reporting

    Returns:
        The processed prompt tokens

    Raises:
        ValueError: If cache wrapper is not initialized or invalid configuration
        ImportError: If mlx dependencies are not available
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "mlx dependencies are not available for text-only prompt processing"
        )

    # Input validation
    if cache_wrapper is None:
        raise ValueError("Cache wrapper is not initialized, cannot process prompt")

    if prompt_tokens is None:
        raise ValueError("prompt_tokens cannot be None")

    if not isinstance(generate_args, dict):
        raise TypeError(f"generate_args must be dict, got {type(generate_args)}")

    # Make sure cache's draft model setting aligns with speculative decoding toggle
    should_use_draft_model = (
        speculative_decoding_toggle
        if speculative_decoding_toggle is not None
        else draft_model is not None
    )

    try:
        if should_use_draft_model:
            if not draft_model:
                raise ValueError(
                    "Speculative decoding toggle is enabled for prompt processing but no "
                    "draft model is loaded"
                )
            cache_wrapper.set_draft_model(draft_model)
            logger.debug("Draft model set for speculative decoding")
        else:
            cache_wrapper.unset_draft_model()
            logger.debug("Draft model unset for standard decoding")

        # Check for common tokens with the previous cache and re-use the cache if possible
        processed_tokens = cache_wrapper.update_cache(
            prompt_tokens,
            prompt_progress_callback,
        )
        generate_args["prompt_cache"] = cache_wrapper.cache

        logger.info(
            f"Text-only prompt processed successfully: {len(processed_tokens)} tokens"
        )
        return processed_tokens

    except Exception as e:
        logger.error(f"Error during text-only prompt processing: {e}")
        # Re-raise with more context
        raise RuntimeError(f"Failed to process text-only prompt: {e}") from e
