import json

import mlx.core as mx
from mlx_vlm.models import qwen3_5, qwen3_5_moe, qwen3_vl, qwen3_vl_moe

from mlx_engine.model_kit.vision_add_ons import qwen3_5 as qwen3_5_addon
from mlx_engine.model_kit.vision_add_ons import load_utils
from mlx_engine.model_kit.vision_add_ons.mistral3 import _sanitize_mistral3_split_weights
from mlx_engine.model_kit.vision_add_ons.qwen3_5 import sanitize_qwen3_5_key


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


def test_load_and_filter_weights_can_transform_qwen3_5_keys_before_filtering(
    monkeypatch,
    tmp_path,
):
    weight_file = tmp_path / "weights.safetensors"
    weight_file.touch()
    weights = {
        "model.language_model.visual.blocks.0.attn.qkv.weight": mx.array([1]),
        "model.visual.patch_embed.proj.weight": mx.array([2]),
        "model.language_model.layers.0.weight": mx.array([3]),
        "lm_head.weight": mx.array([4]),
    }

    monkeypatch.setattr(load_utils.mx, "load", lambda _path: weights)
    components = _FakeComponents()

    filtered = load_utils.load_and_filter_weights(
        tmp_path,
        components,
        weight_key_transformer=sanitize_qwen3_5_key,
    )

    assert set(filtered) == {
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.patch_embed.proj.weight",
    }


def test_qwen3_5_init_common_passes_weight_key_transformer(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_load_vision_addon(**kwargs):
        captured_kwargs.update(kwargs)
        return object(), None, object(), object()

    monkeypatch.setattr(qwen3_5_addon, "load_vision_addon", fake_load_vision_addon)
    addon = qwen3_5_addon.Qwen3_5VisionAddOn.__new__(
        qwen3_5_addon.Qwen3_5VisionAddOn
    )

    addon._init_common(
        model_path=tmp_path,
        model_cls=object,
        model_config_class=object,
        vision_config_class=object,
        text_config_class=object,
        vision_tower_class=object,
        addon_logger=object(),
    )

    assert captured_kwargs["weight_key_transformer"] is sanitize_qwen3_5_key


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


def test_load_and_parse_config_accepts_pre_deserialized_qwen_nested_configs(
    tmp_path,
):
    cases = (
        (
            qwen3_vl,
            {
                "model_type": "qwen3_vl",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rope_theta": 10000,
                "max_position_embeddings": 128,
            },
        ),
        (
            qwen3_vl_moe,
            {
                "model_type": "qwen3_vl_moe",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rope_theta": 10000,
                "max_position_embeddings": 128,
            },
        ),
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
        model_dir = tmp_path / module.__name__.rsplit(".", 1)[-1]
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": module.__name__.rsplit(".", 1)[-1],
                    "text_config": text_config,
                    "vision_config": {"patch_size": 16},
                }
            )
        )

        config, config_dict = load_utils.load_and_parse_config(
            model_dir,
            module.ModelConfig,
            module.VisionConfig,
            module.TextConfig,
        )

        assert isinstance(config.text_config, module.TextConfig)
        assert isinstance(config.vision_config, module.VisionConfig)
        assert config.vision_config.patch_size == 16
        assert config_dict["vision_config"]["patch_size"] == 16
