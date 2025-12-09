"""
Patch management system for automatic model compatibility.

This module provides a centralized system for applying compatibility patches
to various model types transparently during model loading.
"""

import logging
from pathlib import Path
from typing import Dict, Callable, Optional

logger = logging.getLogger(__name__)


class PatchRegistry:
    """
    Registry for managing model-specific compatibility patches.
    """

    def __init__(self):
        self._patches: Dict[str, Callable] = {}
        self._applied_patches: set = set()

    def register(self, model_type: str, patch_func: Callable) -> None:
        """
        Register a patch function for a specific model type.

        Args:
            model_type: The model type identifier (e.g., 'ernie_4_5', 'gemma3n')
            patch_func: Function that applies the compatibility patches
        """
        self._patches[model_type] = patch_func
        logger.debug(f"Registered patch for model type: {model_type}")

    def apply_patches(self, model_type: str) -> None:
        """
        Apply patches for a specific model type if not already applied.

        Args:
            model_type: The model type identifier
        """
        if model_type in self._applied_patches:
            logger.debug(f"Patches already applied for model type: {model_type}")
            return

        if model_type in self._patches:
            logger.info(f"Applying compatibility patches for model type: {model_type}")
            try:
                self._patches[model_type]()
                self._applied_patches.add(model_type)
                logger.info(
                    f"Successfully applied patches for model type: {model_type}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to apply patches for model type {model_type}: {e}"
                )
                raise
        else:
            logger.debug(f"No patches registered for model type: {model_type}")

    def apply_patches_by_config(self, config_path: Path) -> None:
        """
        Apply patches based on model configuration.

        Args:
            config_path: Path to the model's config.json file
        """
        try:
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

            model_type = config.get("model_type")
            if model_type:
                self.apply_patches(model_type)
            else:
                logger.debug("No model_type found in config")
        except Exception as e:
            logger.error(f"Failed to read config from {config_path}: {e}")

    def is_patch_applied(self, model_type: str) -> bool:
        """
        Check if patches have been applied for a model type.

        Args:
            model_type: The model type identifier

        Returns:
            True if patches have been applied, False otherwise
        """
        return model_type in self._applied_patches


# Global patch registry instance
_patch_registry = PatchRegistry()


def get_patch_registry() -> PatchRegistry:
    """Get the global patch registry instance."""
    return _patch_registry


def register_patch(model_type: str, patch_func: Callable) -> None:
    """
    Register a patch function for a specific model type.

    Args:
        model_type: The model type identifier
        patch_func: Function that applies the compatibility patches
    """
    _patch_registry.register(model_type, patch_func)


def apply_patches_for_model(model_type: str) -> None:
    """
    Apply patches for a specific model type.

    Args:
        model_type: The model type identifier
    """
    _patch_registry.apply_patches(model_type)


def apply_patches_by_config(config_path: Path) -> None:
    """
    Apply patches based on model configuration.

    Args:
        config_path: Path to the model's config.json file
    """
    _patch_registry.apply_patches_by_config(config_path)


# Auto-register available patches
def _auto_register_patches():
    """Auto-register all available patches."""
    try:
        from ..model_kit.patches.ernie_4_5 import apply_patches as apply_ernie_patches

        register_patch("ernie_4_5", apply_ernie_patches)
        register_patch("ernie_4_5_moe", apply_ernie_patches)
    except ImportError:
        logger.debug("ERNIE patches not available")

    try:
        from ..model_kit.patches.gemma3n import apply_patches as apply_gemma3n_patches

        register_patch("gemma3n", apply_gemma3n_patches)
    except ImportError:
        logger.debug("Gemma3n patches not available")


# Auto-register patches on module import
_auto_register_patches()
