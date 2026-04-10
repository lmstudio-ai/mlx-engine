from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import (
    BaseVisionAddOn,
)
from mlx_engine.model_kit.vision_add_ons.vision_feature_memoizer import (
    VisionFeatureMemoizer,
)
from mlx_engine.model_kit.vision_add_ons.gemma4 import Gemma4VisionAddOn


class _CountingModule:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


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


def test_gemma4_add_on_reuses_cached_image_features():
    add_on = Gemma4VisionAddOn.__new__(Gemma4VisionAddOn)
    BaseVisionAddOn.__init__(add_on)
    add_on.config = SimpleNamespace(image_token_id=7, audio_token_id=8)
    add_on.processor = object()
    add_on.vision_tower = _CountingModule(
        mx.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=mx.float32)
    )
    add_on.embed_vision = _CountingModule(
        mx.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=mx.float32)
    )

    text_model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                embed_scale=1.0,
                hidden_size_per_layer_input=False,
                embed_tokens=lambda input_ids: mx.stack(
                    [
                        input_ids.astype(mx.float32),
                        input_ids.astype(mx.float32) + 0.5,
                    ],
                    axis=-1,
                ),
            )
        )
    )
    input_ids = mx.array([[1, 7, 7, 2]], dtype=mx.int32)
    pixel_values = mx.ones((1, 3, 2, 2), dtype=mx.float32)

    with patch(
        "mlx_engine.model_kit.vision_add_ons.gemma4.common_process_prompt_with_images",
        return_value=(input_ids, pixel_values, None, None),
    ):
        first_input_ids, first_embeddings = add_on.compute_embeddings(
            text_model,
            prompt_tokens=mx.array([1, 2], dtype=mx.int32),
            images_b64=["img-a"],
            max_size=(512, 512),
        )
        second_input_ids, second_embeddings = add_on.compute_embeddings(
            text_model,
            prompt_tokens=mx.array([3, 4], dtype=mx.int32),
            images_b64=["img-a"],
            max_size=(512, 512),
        )

    assert add_on.vision_tower.calls == 1
    assert add_on.embed_vision.calls == 1
    assert mx.array_equal(first_input_ids, input_ids.squeeze(0))
    assert mx.array_equal(second_input_ids, input_ids.squeeze(0))
    assert mx.array_equal(first_embeddings, second_embeddings)
