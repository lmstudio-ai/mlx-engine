import hashlib
from collections.abc import Callable

import mlx.core as mx
from mlx_vlm.vision_cache import VisionFeatureCache


# Match mlx-vlm's default cache size.
DEFAULT_VISION_FEATURE_CACHE_SIZE = 20


class VisionFeatureMemoizer:
    def __init__(self):
        self._cache = VisionFeatureCache(max_size=DEFAULT_VISION_FEATURE_CACHE_SIZE)

    @property
    def cache(self) -> VisionFeatureCache:
        return self._cache

    def get_or_compute(
        self,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
        compute_features: Callable[[], mx.array],
    ) -> mx.array:
        """Return cached image features or compute, materialize, and cache them."""
        return self.get_or_compute_key(
            self.build_key(images_b64, max_size),
            compute_features,
        )

    def get_or_compute_key(
        self,
        cache_key: str,
        compute_features: Callable[..., mx.array],
        *args,
        **kwargs,
    ) -> mx.array:
        """Return cached image features for an already prepared image identity."""
        features = self._cache.get(cache_key)
        if features is not None:
            return features

        features = compute_features(*args, **kwargs)
        mx.eval(features)
        self._cache.put(cache_key, features)
        return features

    def build_key(
        self,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> str:
        return self._build_key(images_b64, max_size)

    def clear(self) -> None:
        self._cache.clear()

    def _build_key(
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
