"""
Gemma3n compatibility patches using derive and override pattern.

This module provides derived classes that inherit from the original mlx-lm classes
and override specific methods to handle compatibility issues between mlx-vlm and mlx-lm.
"""

import inspect

from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.gemma3n import Model, TextConfig


class CompatibleTextConfig(TextConfig):
    """
    TextConfig that handles intermediate_size as list or integer.

    mlx-vlm's conversion (transformers under the hood) changes the
    "text_config" -> "intermediate_size" value from a single integer to
    a list of integers of length number of layers.
    mlx-lm's model loader expects it to be a single integer.
    This class handles both formats by taking the first value if it's a list.
    """

    @classmethod
    def from_dict(cls, params):
        config_dict = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        if "intermediate_size" in config_dict and isinstance(
            config_dict["intermediate_size"], list
        ):
            config_dict["intermediate_size"] = config_dict["intermediate_size"][0]
        return cls(**config_dict)


class CompatibleModel(Model):
    """
    Model that handles mlx-vlm compatible weight ordering.

    mlx-vlm's conversion changes the weight keys from the original huggingface weights.
    For example, "model.language_model.embed_tokens.weight" becomes
    "language_model.model.embed_tokens.weight".
    mlx-lm expects the weight keys to be in the original huggingface order.
    This class handles both weight formats.
    """

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        if weights.get("language_model", {}).get("model", None) is not None:
            weights = {"model": {"language_model": weights["language_model"]["model"]}}
        weights = dict(tree_flatten(weights))
        return super().sanitize(weights)


def apply_patches():
    """
    Apply gemma3n compatibility patches by replacing classes in the mlx_lm module.
    """
    import mlx_lm.models.gemma3n

    mlx_lm.models.gemma3n.Model = CompatibleModel
    mlx_lm.models.gemma3n.TextConfig = CompatibleTextConfig
