import threading
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.gemma4 import Gemma4VisionAddOn
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import (
    compute_qwen_vl_embeddings,
)


def _token_embeddings(input_ids: mx.array) -> mx.array:
    values = input_ids.astype(mx.float32)
    return mx.stack([values, values + 0.5], axis=-1)


class _FakeGemma4TextModel:
    embed_scale = 1.0
    hidden_size_per_layer_input = False

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return _token_embeddings(input_ids)


class _CountingModule:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


class _MinimalVisionAddOn(BaseVisionAddOn):
    def compute_embeddings(self, *args, **kwargs):
        raise NotImplementedError


def test_base_helper_computes_once_and_reuses_cached_features():
    add_on = _MinimalVisionAddOn()
    compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))

    first = add_on.get_or_compute_cached_vision_features(["img-a"], (512, 512), compute)
    second = add_on.get_or_compute_cached_vision_features(
        ["img-a"], (512, 512), compute
    )

    assert compute.calls == 1
    assert mx.array_equal(first, second)


def test_cache_key_distinguishes_image_order():
    add_on = _MinimalVisionAddOn()
    first_compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))
    second_compute = _CountingModule(mx.array([[3.0, 4.0]], dtype=mx.float32))

    first = add_on.get_or_compute_cached_vision_features(
        ["img-a", "img-b"], (512, 512), first_compute
    )
    second = add_on.get_or_compute_cached_vision_features(
        ["img-b", "img-a"], (512, 512), second_compute
    )

    assert first_compute.calls == 1
    assert second_compute.calls == 1
    assert mx.array_equal(first, mx.array([[1.0, 2.0]], dtype=mx.float32))
    assert mx.array_equal(second, mx.array([[3.0, 4.0]], dtype=mx.float32))


def test_cache_key_distinguishes_max_size():
    add_on = _MinimalVisionAddOn()
    resized_compute = _CountingModule(mx.array([[1.0, 2.0]], dtype=mx.float32))
    larger_compute = _CountingModule(mx.array([[3.0, 4.0]], dtype=mx.float32))
    original_compute = _CountingModule(mx.array([[5.0, 6.0]], dtype=mx.float32))

    resized = add_on.get_or_compute_cached_vision_features(
        ["img-a"], (512, 512), resized_compute
    )
    larger = add_on.get_or_compute_cached_vision_features(
        ["img-a"], (1024, 1024), larger_compute
    )
    original = add_on.get_or_compute_cached_vision_features(
        ["img-a"], None, original_compute
    )

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
        language_model=SimpleNamespace(model=_FakeGemma4TextModel())
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


class _FakeQwenVisionTower:
    def __init__(self, hidden_states: mx.array):
        self.hidden_states = hidden_states
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self.hidden_states, None


class _FakeQwenModel:
    @staticmethod
    def merge_input_ids_with_image_features(
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        image_token_id: int,
        video_token_id: int,
    ) -> tuple[mx.array, mx.array]:
        special_mask = (input_ids == image_token_id) | (input_ids == video_token_id)
        expanded_mask = mx.broadcast_to(special_mask[..., None], inputs_embeds.shape)
        flat_mask = expanded_mask.flatten()
        flat_indices = mx.cumsum(flat_mask.astype(mx.int32)) - 1
        aligned_features = image_features.flatten()[flat_indices % image_features.size]
        merged = mx.where(
            flat_mask,
            aligned_features,
            inputs_embeds.flatten(),
        ).reshape(inputs_embeds.shape)
        return merged, expanded_mask


class _FakeQwenProcessor:
    def decode(self, tokens: list[int]) -> str:
        return f"prompt-{tokens[0]}"


class _FakeQwenAddOn(BaseVisionAddOn):
    def __init__(self, hidden_states: mx.array):
        super().__init__()
        self.processor = _FakeQwenProcessor()
        self.config = SimpleNamespace(image_token_id=99, video_token_id=100)
        self.model_cls = _FakeQwenModel
        self.vision_tower = _FakeQwenVisionTower(hidden_states)

    def compute_embeddings(self, *args, **kwargs):
        raise NotImplementedError


def test_qwen_helper_reuses_cached_image_features_across_prompt_changes():
    add_on = _FakeQwenAddOn(mx.array([[10.0, 11.0], [12.0, 13.0]], dtype=mx.float32))
    text_model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=_token_embeddings)
        )
    )

    def fake_prepare_inputs(*, prompts, **_kwargs):
        prompt_to_ids = {
            "prompt-1": mx.array([[1, 99, 99, 2]], dtype=mx.int32),
            "prompt-2": mx.array([[5, 99, 99, 6]], dtype=mx.int32),
        }
        return {
            "input_ids": prompt_to_ids[prompts],
            "pixel_values": mx.ones((1, 2, 2), dtype=mx.float32),
            "image_grid_thw": mx.array([1, 2, 2], dtype=mx.int32),
        }

    with (
        patch(
            "mlx_engine.model_kit.vision_add_ons.qwen_vl_utils.convert_to_pil",
            return_value=["pil-image"],
        ),
        patch(
            "mlx_engine.model_kit.vision_add_ons.qwen_vl_utils.custom_resize",
            side_effect=lambda images, **_kwargs: images,
        ),
        patch(
            "mlx_engine.model_kit.vision_add_ons.qwen_vl_utils.prepare_inputs",
            side_effect=fake_prepare_inputs,
        ),
    ):
        first = compute_qwen_vl_embeddings(
            addon=add_on,
            text_model=text_model,
            prompt_tokens=mx.array([1], dtype=mx.int32),
            images_b64=["img-a"],
            qwen_vl_version=3,
            max_size=(1024, 1024),
        )
        second = compute_qwen_vl_embeddings(
            addon=add_on,
            text_model=text_model,
            prompt_tokens=mx.array([2], dtype=mx.int32),
            images_b64=["img-a"],
            qwen_vl_version=3,
            max_size=(1024, 1024),
        )

    assert add_on.vision_tower.calls == 1
    assert first.input_ids.tolist() == [1, 99, 99, 2]
    assert second.input_ids.tolist() == [5, 99, 99, 6]

    second_embeddings = second.embeddings.tolist()
    assert second_embeddings[0] == [5.0, 5.5]
    assert second_embeddings[1] == [10.0, 11.0]
    assert second_embeddings[2] == [12.0, 13.0]
    assert second_embeddings[3] == [6.0, 6.5]


class _TrackingVisionAddOn(BaseVisionAddOn):
    def __init__(self):
        super().__init__()
        self.cleared = 0

    def compute_embeddings(self, *args, **kwargs):
        raise NotImplementedError

    def clear_feature_cache(self) -> None:
        self.cleared += 1
        super().clear_feature_cache()


def test_model_kit_shutdown_clears_cached_image_features():
    add_on = _TrackingVisionAddOn()
    first_compute = _CountingModule(mx.ones((1, 2, 2)))
    second_compute = _CountingModule(mx.zeros((1, 2, 2)))

    add_on.get_or_compute_cached_vision_features(["img-a"], (256, 256), first_compute)

    kit = ModelKit.__new__(ModelKit)
    kit.vision_add_on = add_on
    kit._shutdown = threading.Event()

    ModelKit.shutdown(kit)

    result = add_on.get_or_compute_cached_vision_features(
        ["img-a"], (256, 256), second_compute
    )

    assert add_on.cleared == 1
    assert kit.is_shutdown() is True
    assert first_compute.calls == 1
    assert second_compute.calls == 1
    assert mx.array_equal(result, mx.zeros((1, 2, 2)))
