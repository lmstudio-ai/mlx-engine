from typing import Optional

import mlx.nn as nn

from mlx_engine.logging import log_info
from mlx_engine.model_kit import ModelKit


def determine_draft_model_for_generation(
    model_kit: ModelKit, speculative_decoding_toggle: Optional[bool]
) -> Optional[nn.Module]:
    """
    Based on ModelKit and speculative_decoding_toggle, determine draft model to use for
    generation, or None
    """
    if speculative_decoding_toggle is None:
        # toggle not set, use draft model if available
        return model_kit.draft_model
    elif speculative_decoding_toggle and model_kit.draft_model is None:
        raise ValueError(
            "Speculative decoding toggle is explicitly enabled but no draft model is loaded"
        )
    elif not speculative_decoding_toggle and model_kit.draft_model is not None:
        log_info(
            "Draft model is loaded but speculative decoding is disabled for this generation"
        )
        return None
    else:
        # toggle set to true, draft model available
        return model_kit.draft_model


def configure_num_draft_tokens_in_generate_args(
    model_kit: ModelKit,
    draft_model: Optional[nn.Module],
    num_draft_tokens: Optional[int],
    generate_args: dict,
):
    """
    Modifies generate_args in place to include num_draft_tokens if applicable
    """
    if num_draft_tokens is not None:
        if type(model_kit) is not ModelKit:
            log_info(
                message=f"num_draft_tokens setting '{num_draft_tokens}' ignored, "
                f"model_kit (type {type(model_kit).__name__}) must be a text ModelKit instance"
            )
        elif draft_model is None:
            log_info(
                message=f"num_draft_tokens setting '{num_draft_tokens}' ignored, "
                "no draft model loaded/activated for this generation"
            )
        else:
            generate_args["num_draft_tokens"] = num_draft_tokens
