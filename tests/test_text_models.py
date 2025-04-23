import unittest
from pathlib import Path
from utils import model_getter, read_text_file
from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
)


class TestTextModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.model_path_prefix = Path("~/.cache/lm-studio/models").expanduser().resolve()

    def test_prompt_caching_gemma(self):
        model_path = model_getter("mlx-community/gemma-3-text-4b-it-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)
        file_content = read_text_file("data/ben_franklin_autobiography_start.txt")
        prompt = f"""<bos><start_of_turn>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<end_of_turn>
<start_of_turn>model
"""
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        prompt_progress_callback_times_called = 0

        def prompt_progress_callback(progress: float) -> None:
            nonlocal prompt_progress_callback_times_called
            prompt_progress_callback_times_called += 1
            print(f"Prompt Progress: {progress:.2f}")

        def generate() -> None:
            nonlocal generated_text
            for result in create_generator(
                    model_kit=model_kit,
                    prompt_tokens=prompt_tokens,
                    seed=0,
                    max_tokens=10,
                    temp=0.0,
                    prompt_progress_callback=prompt_progress_callback,
            ):
                print(result.text, end="", flush=True)
                generated_text += result.text
                if result.stop_condition:
                    break

        generate()
        self.assertEqual(prompt_progress_callback_times_called, 8)
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        ben_franklin_in_response = "Benjamin Franklin" in generated_text
        self.assertTrue(
            ben_franklin_in_response,
            f"Model failed to identify Ben Franklin. Generated: '{generated_text}'",
        )

        prompt += generated_text
        prompt += """<end_of_turn>
<start_of_turn>user
repeat<end_of_turn>
<start_of_turn>model
"""
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        prompt_progress_callback_times_called = 0

        generate()

        self.assertEqual(prompt_progress_callback_times_called, 0)
