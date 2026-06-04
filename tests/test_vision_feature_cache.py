import mlx.core as mx

from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    PreparedPrompt,
    build_prompt_kwargs,
)
from mlx_engine.model_kit.batched_vision.vision_feature_memoizer import (
    VisionFeatureMemoizer,
)


class _CountingModule:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


class _EmbeddingOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_dict(self):
        return dict(self.kwargs)


def test_feature_memoizer_computes_once_and_reuses_cached_features():
    memoizer = VisionFeatureMemoizer()
    compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))

    first = memoizer.get_or_compute(["img-a"], (512, 512), compute)
    second = memoizer.get_or_compute(["img-a"], (512, 512), compute)

    assert compute.calls == 1
    assert mx.array_equal(first, second)


def test_feature_memoizer_keys_image_order():
    memoizer = VisionFeatureMemoizer()
    first_compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))
    second_compute = _CountingModule(mx.array([[3.0, 4.0]], dtype=mx.float32))

    first = memoizer.get_or_compute(["img-a", "img-b"], (512, 512), first_compute)
    second = memoizer.get_or_compute(["img-b", "img-a"], (512, 512), second_compute)

    assert first_compute.calls == 1
    assert second_compute.calls == 1
    assert mx.array_equal(first, mx.array([[1.0, 2.0]], dtype=mx.float32))
    assert mx.array_equal(second, mx.array([[3.0, 4.0]], dtype=mx.float32))


def test_feature_memoizer_keys_max_size():
    memoizer = VisionFeatureMemoizer()
    resized_compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))
    larger_compute = _CountingModule(mx.array([[3.0, 4.0]], dtype=mx.float32))
    original_compute = _CountingModule(mx.array([[5.0, 6.0]], dtype=mx.float32))

    resized = memoizer.get_or_compute(["img-a"], (512, 512), resized_compute)
    larger = memoizer.get_or_compute(["img-a"], (1024, 1024), larger_compute)
    original = memoizer.get_or_compute(["img-a"], None, original_compute)

    assert resized_compute.calls == 1
    assert larger_compute.calls == 1
    assert original_compute.calls == 1
    assert mx.array_equal(resized, mx.array([[1.0, 2.0]], dtype=mx.float32))
    assert mx.array_equal(larger, mx.array([[3.0, 4.0]], dtype=mx.float32))
    assert mx.array_equal(original, mx.array([[5.0, 6.0]], dtype=mx.float32))


class _BatchedEncodeImageModel:
    def __init__(self):
        self.encode_image = _CountingModule(
            mx.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=mx.float32)
        )
        self.embedding_calls = []

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        self.embedding_calls.append(kwargs)
        image_features = kwargs["cached_image_features"]
        return _EmbeddingOutput(inputs_embeds=image_features)


def test_batched_prompt_kwargs_reuses_cached_encoded_image_features():
    memoizer = VisionFeatureMemoizer()
    model = _BatchedEncodeImageModel()
    raw_inputs = {
        "input_ids": mx.array([[1, 7, 7, 2]], dtype=mx.int32),
        "pixel_values": mx.ones((1, 3, 2, 2), dtype=mx.float32),
        "attention_mask": mx.array([[1, 1, 1, 1]], dtype=mx.int32),
    }
    prepared_prompt = PreparedPrompt(
        prompt_input_ids=[1, 7, 7, 2],
        raw_inputs=raw_inputs,
        image_spans=[PromptImageSpan(1, 3, "image-a")],
        vision_cache_key="prepared-images:image-a",
    )

    first = build_prompt_kwargs(model, prepared_prompt, memoizer)
    second = build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert model.encode_image.calls == 1
    assert mx.array_equal(first["inputs_embeds"], second["inputs_embeds"])
    assert "cached_image_features" not in first
    assert "vision_cache" not in first
    assert "_image_key" not in first


class _InternalVisionCacheModel:
    def __init__(self):
        self.vision_calls = 0

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        vision_cache = kwargs["vision_cache"]
        image_key = kwargs["_image_key"]
        features = vision_cache.get(image_key)
        if features is None:
            self.vision_calls += 1
            features = mx.array([[[20.0, 21.0]]], dtype=mx.float32)
            mx.eval(features)
            vision_cache.put(image_key, features)
        return _EmbeddingOutput(inputs_embeds=features)


def test_batched_prompt_kwargs_passes_internal_vision_cache_without_leaking_kwargs():
    memoizer = VisionFeatureMemoizer()
    model = _InternalVisionCacheModel()
    prepared_prompt = PreparedPrompt(
        prompt_input_ids=[1, 7, 2],
        raw_inputs={
            "input_ids": mx.array([[1, 7, 2]], dtype=mx.int32),
            "pixel_values": mx.ones((1, 3, 2, 2), dtype=mx.float32),
        },
        image_spans=[PromptImageSpan(1, 2, "image-a")],
        vision_cache_key="prepared-images:image-a",
    )

    first = build_prompt_kwargs(model, prepared_prompt, memoizer)
    second = build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert model.vision_calls == 1
    assert mx.array_equal(first["inputs_embeds"], second["inputs_embeds"])
    assert "vision_cache" not in first
    assert "_image_key" not in first


class _EncodeImageWithPositionIdsModel:
    def __init__(self):
        self.encode_calls = []

    def encode_image(self, pixel_values, *, image_position_ids=None):
        self.encode_calls.append((pixel_values, image_position_ids))
        return mx.array([[[30.0, 31.0]]], dtype=mx.float32)

    def get_input_embeddings(self, input_ids, pixel_values, mask=None, **kwargs):
        return _EmbeddingOutput(inputs_embeds=kwargs["cached_image_features"])


def test_batched_prompt_kwargs_passes_position_ids_to_encode_image():
    memoizer = VisionFeatureMemoizer()
    model = _EncodeImageWithPositionIdsModel()
    image_position_ids = mx.array([[[0, 1], [1, -1]]], dtype=mx.int32)
    prepared_prompt = PreparedPrompt(
        prompt_input_ids=[1, 7, 2],
        raw_inputs={
            "input_ids": mx.array([[1, 7, 2]], dtype=mx.int32),
            "pixel_values": mx.ones((1, 3, 2, 2), dtype=mx.float32),
            "image_position_ids": image_position_ids,
        },
        image_spans=[PromptImageSpan(1, 2, "image-a")],
        vision_cache_key="prepared-images:image-a",
    )

    build_prompt_kwargs(model, prepared_prompt, memoizer)
    build_prompt_kwargs(model, prepared_prompt, memoizer)

    assert len(model.encode_calls) == 1
    assert model.encode_calls[0][1] is image_position_ids
