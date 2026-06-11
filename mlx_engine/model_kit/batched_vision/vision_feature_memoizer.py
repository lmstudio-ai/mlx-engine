from mlx_vlm.vision_cache import VisionFeatureCache


# Match mlx-vlm's default cache size.
DEFAULT_VISION_FEATURE_CACHE_SIZE = 20


class VisionFeatureMemoizer:
    def __init__(self, max_size: int = DEFAULT_VISION_FEATURE_CACHE_SIZE):
        self._cache = VisionFeatureCache(max_size=max_size)

    @property
    def cache(self) -> VisionFeatureCache:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()
