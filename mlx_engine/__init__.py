"""
`mlx_engine` is LM Studio's LLM inferencing engine for Apple MLX
"""

__all__ = [
    "load_model",
    "get_runtime_load_info",
    "load_draft_model",
    "is_draft_model_compatible",
    "unload_draft_model",
    "create_generator",
    "stop_generation",
    "tokenize",
    "unload",
]

from pathlib import Path
import os

# Outlines can open its cache during import, so configure it before loading dependencies.
_lmstudio_home = os.environ.get("LMS_LMSTUDIO_HOME")
if _lmstudio_home and not os.environ.get("OUTLINES_CACHE_DIR"):
    os.environ["OUTLINES_CACHE_DIR"] = str(
        Path(_lmstudio_home) / ".internal" / "outlines"
    )

from .utils.disable_hf_download import patch_huggingface_hub
from .utils.register_models import register_models
from .utils.logger import setup_logging


from .generate import (
    load_model,
    get_runtime_load_info,
    load_draft_model,
    is_draft_model_compatible,
    unload_draft_model,
    create_generator,
    tokenize,
    unload,
    stop_generation,
)

patch_huggingface_hub()
register_models()
setup_logging()
