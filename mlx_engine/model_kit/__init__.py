"""
Model Kit module with automatic compatibility patches.

This module automatically applies compatibility patches for various model types
by replacing classes in their respective modules with derived, compatible versions.

Patches are applied transparently during model loading without requiring user intervention.
"""

# Import the main ModelKit class
from .model_kit import ModelKit

__all__ = ["ModelKit"]
