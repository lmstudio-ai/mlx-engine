from typing import Optional, List, Callable

from mlx import nn
import mlx.core as mx

from mlx_engine.cache_wrapper import CacheWrapper


def process_prompt_text_only(
    prompt_tokens: mx.array,
    cache_wrapper: CacheWrapper,
    generate_args: dict = None,
    draft_model: Optional[nn.Module] = None,
    speculative_decoding_toggle: Optional[bool] = None,
    prompt_progress_callback: Optional[Callable[[float], None]] = None,
):
    if cache_wrapper is None:
        raise ValueError("Cache wrapper is not initialized, cannot process prompt")
    if generate_args is None:
        generate_args = {}
    # Make sure cache's draft model setting aligns with speculative decoding toggle
    should_use_draft_model = (
        speculative_decoding_toggle
        if speculative_decoding_toggle is not None
        else draft_model is not None
    )
    if should_use_draft_model:
        if not draft_model:
            raise ValueError(
                "Speculative decoding toggle is enabled for prompt processing but no "
                "draft model is loaded"
            )
        cache_wrapper.set_draft_model(draft_model)
    else:
        cache_wrapper.unset_draft_model()

    # Check for common tokens with the previous cache and re-use the cache if possible
    prompt_tokens = cache_wrapper.update_cache(
        prompt_tokens,
        prompt_progress_callback,
    )
    generate_args["prompt_cache"] = cache_wrapper.cache
    return prompt_tokens
