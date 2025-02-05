import unittest
from pathlib import Path
from .utils import model_getter
from mlx_engine.generate import (
    load_model,
    load_draft_model,
    is_draft_model_compatible,
    unload_draft_model,
    tokenize,
    create_generator,
)


class TestSpeculativeDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.model_path_prefix = Path("~/.cache/lm-studio/models").expanduser().resolve()

    def test_is_draft_model_compatible_true_vocab_only_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, vocab_only=True)
        self.assertTrue(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_true_full_model_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path)
        self.assertTrue(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_false_vocab_only_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, vocab_only=True)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_false_full_model_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_false_vision(self):
        model_path = model_getter("mlx-community/Qwen2-VL-7B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_load_draft_model_success(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None)
        load_draft_model(model_kit=model_kit, path=draft_model_path)
        self.assertIsNotNone(model_kit.draft_model)

    def test_load_draft_model_invalid_model(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=None)
        with self.assertRaises(ValueError):
            load_draft_model(model_kit=model_kit, path=draft_model_path)

    def test_unload_draft_model_idempotent_none_loaded(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=None)
        unload_draft_model(model_kit=model_kit)

    def test_unload_draft_model_success(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None)
        model_kit.load_draft_model(path=draft_model_path)
        unload_draft_model(model_kit=model_kit)
        self.assertIsNone(model_kit.draft_model)

    def test_basic_generation(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None)
        load_draft_model(model_kit=model_kit, path=draft_model_path)
        prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            seed=0,
            max_tokens=10,
            temp=0.0,
        ):
            generated_text += result.text
            if result.stop_condition:
                break

        # Verify the output
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        paris_in_response = "paris" in generated_text.lower()
        self.assertTrue(
            paris_in_response,
            f"Model failed to respond correctly",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
