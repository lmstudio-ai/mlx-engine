import mlx.core as mx
from mlx_vlm.models import qwen3_5, qwen3_5_moe

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


def test_config_dict_sync_keeps_qwen3_5_quantization_key_sanitization():
    quantization = {
        "group_size": 128,
        "bits": 4,
        "model.language_model.layers.0.linear_attn.in_proj_qkv": False,
        "model.visual.blocks.0.attn.qkv": False,
        "lm_head": False,
    }

    cases = (
        (
            qwen3_5,
            {
                "model_type": "qwen3_5",
                "hidden_size": 8,
                "intermediate_size": 16,
                "linear_num_value_heads": 1,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
            },
        ),
        (
            qwen3_5_moe,
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "linear_num_value_heads": 1,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "shared_expert_intermediate_size": 8,
                "moe_intermediate_size": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
            },
        ),
    )

    for module, text_config in cases:
        config_dict = {
            "model_type": module.__name__.rsplit(".", 1)[-1],
            "text_config": text_config,
            "vision_config": {},
            "quantization": quantization,
            "quantization_config": quantization,
        }
        config = module.ModelConfig.from_dict(config_dict)
        load_utils._sync_config_dict_quantization_fields(config, config_dict)

        assert config_dict["quantization"] is config.quantization
        assert config_dict["quantization_config"] is config.quantization_config
        assert (
            "language_model.model.layers.0.linear_attn.in_proj_qkv"
            in config_dict["quantization"]
        )
        assert "vision_tower.blocks.0.attn.qkv" in config_dict["quantization"]
        assert "language_model.lm_head" in config_dict["quantization"]
        assert (
            "model.language_model.layers.0.linear_attn.in_proj_qkv"
            not in config_dict["quantization"]
        )
        assert "model.visual.blocks.0.attn.qkv" not in config_dict["quantization"]
