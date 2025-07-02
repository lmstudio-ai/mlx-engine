"""
Model Kit module with automatic compatibility patches.

This module automatically applies compatibility patches for various model types
by replacing classes in their respective modules with derived, compatible versions.
"""

from .patches.gemma3n import apply_patches as apply_patches_gemma3n

apply_patches_gemma3n()

# Keep namespace clean
del apply_patches_gemma3n
