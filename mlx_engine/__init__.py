"""
`mlx_engine` is LM Studio's LLM inferencing engine for Apple MLX
"""

from pathlib import Path
import os


def _set_outlines_cache_dir(cache_dir: Path | str):
    """
    Set the cache dir for Outlines.

    Outlines reads the OUTLINES_CACHE_DIR environment variable to
    determine where to read/write its cache files
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    os.environ["OUTLINES_CACHE_DIR"] = str(cache_dir)


_set_outlines_cache_dir(Path("~/.cache/lm-studio/.internal/outlines"))

"""
The API for `mlx_engine` is specified in generate.py
"""
from .generate import load_model, create_generator, tokenize
