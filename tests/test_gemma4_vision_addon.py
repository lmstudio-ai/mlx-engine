from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.gemma4 import _compute_prompt_per_layer_inputs


class _FakeGemma4TextModel:
    def __init__(self, hidden_size_per_layer_input: int = 1):
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.seen_input_ids = None

    def _get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        self.seen_input_ids = input_ids
        return input_ids[..., None, None]


class _FakeGemma4Model:
    def __init__(self, hidden_size_per_layer_input: int = 1):
        self.language_model = SimpleNamespace(
            model=_FakeGemma4TextModel(hidden_size_per_layer_input)
        )


def test_compute_prompt_per_layer_inputs_masks_special_tokens():
    text_model = _FakeGemma4Model()
    input_ids = mx.array([[1, 99, 2, 100, 3]], dtype=mx.int32)

    prompt_per_layer_inputs = _compute_prompt_per_layer_inputs(
        text_model.language_model.model,
        input_ids,
        image_token_id=99,
        audio_token_id=100,
    )

    assert text_model.language_model.model.seen_input_ids.tolist() == [[1, 0, 2, 0, 3]]
    assert prompt_per_layer_inputs.shape == (1, 5, 1, 1)
    assert prompt_per_layer_inputs[0, :, 0, 0].tolist() == [1, 0, 2, 0, 3]


def test_compute_prompt_per_layer_inputs_skips_models_without_per_layer_inputs():
    text_model = _FakeGemma4Model(hidden_size_per_layer_input=0)
    input_ids = mx.array([[1, 99, 2]], dtype=mx.int32)

    assert (
        _compute_prompt_per_layer_inputs(
            text_model.language_model.model,
            input_ids,
            image_token_id=99,
            audio_token_id=100,
        )
        is None
    )
    assert text_model.language_model.model.seen_input_ids is None
