"""
Set the cache dir for outlines.

Outlines reads this environment variable to determine where to read/write
its cache files
"""

from pathlib import Path
import os

OUTLINES_CACHE_DIR = Path("~/.cache/lm-studio/.internal/outlines").expanduser()
os.environ["OUTLINES_CACHE_DIR"] = str(OUTLINES_CACHE_DIR)

"""
The API for `mlx_engine` should be specified in generate.py
"""
from .generate import *
