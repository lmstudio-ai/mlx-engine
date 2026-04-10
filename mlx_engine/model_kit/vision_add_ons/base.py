import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_vlm.vision_cache import VisionFeatureCache


# Match mlx-vlm's default cache size to keep behavior aligned across servers.
DEFAULT_VISION_FEATURE_CACHE_SIZE = 20


class BaseVisionAddOn(ABC):
    """
    Base class that defines the interface for a VisionAddOn.
    """

    def __init__(self):
        """
        Where load of vision model components is intended to occur.
        """
        self._vision_feature_cache = VisionFeatureCache(
            max_size=DEFAULT_VISION_FEATURE_CACHE_SIZE
        )

    @abstractmethod
    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns input ids and input embeddings for the language model after text/image merging of the prompt.

        Args:
            text_model: Text model for embedding tokens
            prompt_tokens: Input prompt tokens
            images_b64: List of base64-encoded images
            max_size: Maximum image size as (width, height) tuple. If None, no resizing.
        """

    def clear_prediction_state(self, text_model: nn.Module) -> None:
        """
        Called before every prediction to reset any model state set by a
        previous request. Default is a no-op; override in add-ons that
        inject state into the text model (e.g., MRoPE positions).
        """

    def get_or_compute_cached_vision_features(
        self,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
        compute_features: Callable[[], Any],
    ) -> Any:
        """Return cached image features or compute, materialize, and cache them."""
        cache_key = self._build_vision_cache_key(images_b64, max_size)
        features = self._vision_feature_cache.get(cache_key)
        if features is not None:
            return features

        features = compute_features()
        mx.eval(features)
        self._vision_feature_cache.put(cache_key, features)
        return features

    def clear_feature_cache(self) -> None:
        """Release cached image features."""
        self._vision_feature_cache.clear()

    def _build_vision_cache_key(
        self,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> str:
        """Keep cache hits exact across image changes, reorderings, and resize settings."""
        size_key = "orig" if max_size is None else f"{max_size[0]}x{max_size[1]}"
        image_hashes = [
            hashlib.sha256(image.encode("utf-8")).hexdigest() for image in images_b64
        ]
        return f"{size_key}:{'|'.join(image_hashes)}"
