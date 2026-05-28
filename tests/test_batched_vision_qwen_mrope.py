import pytest

import mlx.core as mx

from mlx_engine.model_kit.batched_vision.qwen_mrope import (
    apply_qwen_image_mrope_state,
    build_qwen_image_mrope_state,
)


class _FakeLanguageModel:
    pass


class _FakeVisionConfig:
    spatial_merge_size = 1


class _FakeConfig:
    image_token_id = 20
    vision_config = _FakeVisionConfig()


class _FakeModel:
    def __init__(self):
        self.language_model = _FakeLanguageModel()
        self.config = _FakeConfig()


def test_qwen_image_mrope_state_walks_multiple_image_runs():
    state = build_qwen_image_mrope_state(
        input_ids=mx.array([[101, 20, 20, 102, 20, 20, 20, 20, 103, 104]]),
        image_grid_thw=mx.array([[1, 1, 2], [1, 2, 2]]),
        image_token_id=20,
        spatial_merge_size=1,
    )

    assert state.position_ids.tolist() == [
        [[0, 1, 1, 3, 4, 4, 4, 4, 6, 7]],
        [[0, 1, 1, 3, 4, 4, 5, 5, 6, 7]],
        [[0, 1, 2, 3, 4, 5, 4, 5, 6, 7]],
    ]
    assert state.rope_deltas.item() == -2


def test_qwen_image_mrope_state_respects_spatial_merge_size():
    state = build_qwen_image_mrope_state(
        input_ids=mx.array([[1, 20, 20, 20, 20, 2]]),
        image_grid_thw=mx.array([[1, 4, 4]]),
        image_token_id=20,
        spatial_merge_size=2,
    )

    assert state.position_ids.tolist() == [
        [[0, 1, 1, 1, 1, 3]],
        [[0, 1, 1, 2, 2, 3]],
        [[0, 1, 2, 1, 2, 3]],
    ]
    assert state.rope_deltas.item() == -2


def test_qwen_image_mrope_state_rejects_mismatched_image_runs():
    with pytest.raises(ValueError, match="runs vs"):
        build_qwen_image_mrope_state(
            input_ids=mx.array([[101, 20, 20, 102, 20, 20]]),
            image_grid_thw=mx.array([[1, 1, 2]]),
            image_token_id=20,
            spatial_merge_size=1,
        )


def test_apply_qwen_image_mrope_state_updates_mlx_vlm_side_state():
    model = _FakeModel()

    state = apply_qwen_image_mrope_state(
        model,
        input_ids=mx.array([[101, 20, 20, 102]]),
        image_grid_thw=mx.array([[1, 1, 2]]),
    )

    assert state is not None
    assert model.language_model._position_ids is state.position_ids
    assert model.language_model._rope_deltas is state.rope_deltas
