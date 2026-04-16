import hashlib
from collections.abc import Callable

import mlx.core as mx
from mlx_vlm.vision_cache import VisionFeatureCache


# Match mlx-vlm's default cache size
DEFAULT_VISION_FEATURE_CACHE_SIZE = 20


class VisionFeatureMemoizer:
    def __init__(self):
        self._cache = VisionFeatureCache(max_size=DEFAULT_VISION_FEATURE_CACHE_SIZE)

    def get_or_compute(
        self,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
        compute_features: Callable[[], mx.array],
    ) -> mx.array:
        """Return cached image features or compute, materialize, and cache them."""
        cache_key = self._build_key(images_b64, max_size)
        features = self._cache.get(cache_key)
        if features is not None:
            return features

        features = compute_features()
        mx.eval(features)
        self._cache.put(cache_key, features)
        return features

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
