import mlx.core as mx

from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    PreparedPrompt,
    build_prompt_kwargs,
)
from mlx_engine.model_kit.batched_vision.vision_feature_memoizer import (
    VisionFeatureMemoizer,
)


class _EmbeddingOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_dict(self):
        return dict(self.kwargs)


class _InternalVisionCacheModel:
    def __init__(self):
        self.embedding_calls = []
        self.vision_calls = 0

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        self.embedding_calls.append(kwargs)
        vision_cache = kwargs.get("vision_cache")
        image_key = kwargs.get("_image_key")

        features = None
        if vision_cache is not None and image_key is not None:
            features = vision_cache.get(image_key)

        if features is None:
            self.vision_calls += 1
            features = mx.array([[[20.0, 21.0]]], dtype=mx.float32)
            mx.eval(features)
            if vision_cache is not None and image_key is not None:
                vision_cache.put(image_key, features)

        return _EmbeddingOutput(inputs_embeds=features)


class _NestedInternalVisionCacheModel:
    def __init__(self):
        self.embedding_calls = []
        self.vision_calls = 0

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        self.embedding_calls.append(kwargs)

        def cached_features():
            vision_cache = kwargs.get("vision_cache")
            image_key = kwargs.get("_image_key")
            if vision_cache is None or image_key is None:
                return None
            return vision_cache.get(image_key)

        features = cached_features()
        if features is None:
            self.vision_calls += 1
            features = mx.array([[[20.0, 21.0]]], dtype=mx.float32)
            mx.eval(features)
            vision_cache = kwargs.get("vision_cache")
            image_key = kwargs.get("_image_key")
            if vision_cache is not None and image_key is not None:
                vision_cache.put(image_key, features)

        return _EmbeddingOutput(inputs_embeds=features)


class _CachedFeaturesOnlyModel:
    def __init__(self):
        self.embedding_calls = []
        self.vision_calls = 0

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        self.embedding_calls.append(kwargs)
        features = kwargs.get("cached_image_features")
        if features is None:
            self.vision_calls += 1
            features = mx.array([[[20.0, 21.0]]], dtype=mx.float32)
        return _EmbeddingOutput(inputs_embeds=features)


def test_feature_memoizer_owns_upstream_cache():
    memoizer = VisionFeatureMemoizer(max_size=1)
    first = mx.array([[1.0, 2.0]], dtype=mx.float32)
    second = mx.array([[3.0, 4.0]], dtype=mx.float32)

    memoizer.cache.put("image-a", first)
    memoizer.cache.put("image-b", second)

    assert memoizer.cache.get("image-a") is None
    assert mx.array_equal(memoizer.cache.get("image-b"), second)

    memoizer.clear()

    assert len(memoizer.cache) == 0


def test_batched_prompt_kwargs_passes_internal_vision_cache_when_model_uses_image_key():
    memoizer = VisionFeatureMemoizer()
    model = _InternalVisionCacheModel()
    prepared_prompt = _prepared_prompt()

    first = build_prompt_kwargs(model, prepared_prompt, memoizer)
    second = build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert model.vision_calls == 1
    assert mx.array_equal(first["inputs_embeds"], second["inputs_embeds"])
    assert all("vision_cache" in call for call in model.embedding_calls)
    assert all("_image_key" in call for call in model.embedding_calls)
    assert "vision_cache" not in first
    assert "_image_key" not in first


def test_batched_prompt_kwargs_detects_nested_image_key_cache_helpers():
    memoizer = VisionFeatureMemoizer()
    model = _NestedInternalVisionCacheModel()
    prepared_prompt = _prepared_prompt()

    first = build_prompt_kwargs(model, prepared_prompt, memoizer)
    second = build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert model.vision_calls == 1
    assert mx.array_equal(first["inputs_embeds"], second["inputs_embeds"])
    assert all("vision_cache" in call for call in model.embedding_calls)
    assert all("_image_key" in call for call in model.embedding_calls)


def test_batched_prompt_kwargs_passes_cache_kwargs_even_when_model_ignores_them():
    memoizer = VisionFeatureMemoizer()
    model = _CachedFeaturesOnlyModel()
    prepared_prompt = _prepared_prompt()

    first = build_prompt_kwargs(model, prepared_prompt, memoizer)
    second = build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert model.vision_calls == 2
    assert mx.array_equal(first["inputs_embeds"], second["inputs_embeds"])
    assert all("vision_cache" in call for call in model.embedding_calls)
    assert all("_image_key" in call for call in model.embedding_calls)
    assert "vision_cache" not in first
    assert "_image_key" not in first


def _prepared_prompt() -> PreparedPrompt:
    return PreparedPrompt(
        prompt_input_ids=[1, 7, 2],
        raw_inputs={
            "input_ids": mx.array([[1, 7, 2]], dtype=mx.int32),
            "pixel_values": mx.ones((1, 3, 2, 2), dtype=mx.float32),
        },
        image_spans=[PromptImageSpan(1, 2, "image-a")],
        vision_cache_key="prepared-images:image-a",
    )
