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
    "unload",
]

from pathlib import Path
import os

from .utils.disable_hf_download import patch_huggingface_hub
from .utils.register_models import register_models
from .utils.logger import setup_logging


from .generate import (
    load_model,
    load_draft_model,
    is_draft_model_compatible,
    unload_draft_model,
    create_generator,
    tokenize,
    unload,
)

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
