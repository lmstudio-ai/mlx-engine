import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons import load_utils
from mlx_engine.model_kit.vision_add_ons.mistral3 import _sanitize_mistral3_split_weights


class _FakeComponents:
    def __init__(self):
        self.loaded_keys = []
        self.expected_keys = {
            "vision_tower.vision_model.transformer.layers.0.weight",
            "multi_modal_projector.linear.weight",
            "vision_tower.vision_model.patch_conv.weight",
        }

    def children(self):
        return {
            "vision_tower": object(),
            "multi_modal_projector": object(),
        }

    def load_weights(self, weights):
        loaded_keys = [key for key, _value in weights]
        unexpected_keys = set(loaded_keys) - self.expected_keys
        if unexpected_keys:
            raise ValueError(f"unexpected keys: {sorted(unexpected_keys)}")
        self.loaded_keys.extend(loaded_keys)

    def parameters(self):
        return []

    def eval(self):
        pass


def test_load_and_filter_weights_keeps_model_prefixed_component_keys(
    monkeypatch,
    tmp_path,
):
    """Mistral3 sanitize needs these keys after the local pre-filter runs."""
    weight_file = tmp_path / "weights.safetensors"
    weight_file.touch()
    weights = {
        "model.vision_tower.transformer.layers.0.weight": mx.array([1]),
        "model.multi_modal_projector.linear.weight": mx.array([2]),
        "model.language_model.layers.0.weight": mx.array([3]),
        "vision_tower.patch_conv.weight": mx.array([4]),
        "vision_tower_extra.patch_conv.weight": mx.array([5]),
    }

    monkeypatch.setattr(load_utils.mx, "load", lambda _path: weights)
    components = _FakeComponents()

    filtered = load_utils.load_and_filter_weights(tmp_path, components)

    assert set(filtered) == {
        "vision_tower.transformer.layers.0.weight",
        "multi_modal_projector.linear.weight",
        "vision_tower.patch_conv.weight",
    }


def test_mistral3_split_weight_sanitizer_rewrites_prefiltered_keys(
    monkeypatch,
    tmp_path,
):
    weight_file = tmp_path / "weights.safetensors"
    weight_file.touch()
    weights = {
        "model.vision_tower.transformer.layers.0.weight": mx.array([1]),
        "model.multi_modal_projector.linear.weight": mx.array([2]),
        "vision_tower.patch_conv.weight": mx.array([3]),
    }

    monkeypatch.setattr(load_utils.mx, "load", lambda _path: weights)
    monkeypatch.setattr(load_utils.mx, "eval", lambda _parameters: None)
    components = _FakeComponents()

    filtered = load_utils.load_and_filter_weights(tmp_path, components)
    sanitized = _sanitize_mistral3_split_weights(filtered)
    load_utils.prepare_components(components, sanitized)

    assert set(sanitized) == components.expected_keys
    assert components.loaded_keys == [
        "vision_tower.vision_model.transformer.layers.0.weight",
        "multi_modal_projector.linear.weight",
        "vision_tower.vision_model.patch_conv.weight",
    ]
