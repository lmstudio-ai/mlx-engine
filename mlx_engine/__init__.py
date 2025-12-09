"""
`mlx_engine` is LM Studio's LLM inferencing engine for Apple MLX
"""

__all__ = [
    "load_model",
    "load_draft_model",
    "is_draft_model_compatible",
    "unload_draft_model",
    "create_generator",
    "tokenize",
    "cli_parser",
    "select_profile_for_hardware",
]

import os
from pathlib import Path

SKIP_INIT = os.environ.get("MLX_ENGINE_SKIP_INIT") == "1"

from .generate import (
    cli_parser,
    create_generator,
    is_draft_model_compatible,
    load_draft_model,
    load_model,
    tokenize,
    unload_draft_model,
)
from .utils.hardware import (
    select_profile_for_hardware,
)

if not SKIP_INIT:
    from .utils.disable_hf_download import patch_huggingface_hub
    from .utils.logger import setup_logging
    from .utils.register_models import register_models

    patch_huggingface_hub()
    register_models()
    setup_logging()


def _set_outlines_cache_dir(cache_dir: Path | str):
    """
    Set the cache dir for Outlines.

    Outlines reads the OUTLINES_CACHE_DIR environment variable to
    determine where to read/write its cache files
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    os.environ["OUTLINES_CACHE_DIR"] = str(cache_dir)


_set_outlines_cache_dir(Path("~/.cache/lm-studio/.internal/outlines"))
