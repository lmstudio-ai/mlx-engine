import unittest
from pathlib import Path

from mlx_engine.logging import log_info
from .utils import model_getter, read_text_file
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

    def test_repetition_penalty_applies(self):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)
        prompt = """<|im_start|>user
The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. The quick brown fox jumped over the lazy dog. Repeat what I said.
<|im_end|>\n<|im_start|>assistant\n"""
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""

        def generate() -> None:
            nonlocal generated_text
            for result in create_generator(
                    model_kit=model_kit,
                    prompt_tokens=prompt_tokens,
                    repetition_penalty=3, # this will make it so that model shouldn't repeat. If set to 0, it will
                    repetition_context_size=64,
                    seed=0,
                    max_tokens=20,
                    temp=0.0,
            ):
                print(result.text, end="", flush=True)
                generated_text += result.text
                if result.stop_condition:
                    break
            print("\n", flush=True)

        generate()
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        self.assertNotIn("The quick brown fox jumped over the lazy dog.", generated_text)

    def test_prompt_caching(self):
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
            print("\n", flush=True)

        def reset_state() -> None:
            nonlocal prompt_progress_callback_times_called, generated_text
            generated_text = ""
            prompt_progress_callback_times_called = 0

        ### Generation 1
        generate()
        self.assertEqual(prompt_progress_callback_times_called, 4)
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
        reset_state()

        ### Generation 2
        generate()
        # Expect prompt cache to be intact, so we should only get 0%, 100% callbacks and no intermediates
        self.assertEqual(prompt_progress_callback_times_called, 2)
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        ben_franklin_in_response = "Benjamin Franklin" in generated_text
        self.assertTrue(
            ben_franklin_in_response,
            f"Model failed to identify Ben Franklin. Generated: '{generated_text}'",
        )

    def test_prompt_caching_trim_qwen2_5(self):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)
        file_content = read_text_file("data/ben_franklin_autobiography_start.txt")
        prompt = f"""<|im_start|>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<end_of_turn>
<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        log_info(
            prefix="test_prompt_caching_trim",
            message=f"Generation 1 number of prompt tokens: {len(prompt_tokens)}",
        )
        generated_text_list_1 = []
        prompt_progress_callback_times_called = 0

        def prompt_progress_callback(progress: float) -> None:
            nonlocal prompt_progress_callback_times_called
            prompt_progress_callback_times_called += 1
            print(f"Prompt Progress: {progress:.2f}")

        # accumulating to list allows pass by reference
        def generate(text_accumulator: list) -> None:
            for result in create_generator(
                    model_kit=model_kit,
                    prompt_tokens=prompt_tokens,
                    seed=0,
                    max_tokens=10,
                    temp=0.0,
                    prompt_progress_callback=prompt_progress_callback,
            ):
                print(result.text, end="", flush=True)
                text_accumulator.append(result.text)
                if result.stop_condition:
                    break
            print("\n", flush=True)

        ### Generation 1 - fills cache
        generate(text_accumulator=generated_text_list_1)
        generated_text_1 = "".join(generated_text_list_1)
        self.assertEqual(prompt_progress_callback_times_called, 4)
        self.assertGreater(len(generated_text_1), 0, "Model failed to generate any text")
        ben_franklin_in_response = "Benjamin Franklin" in generated_text_1
        self.assertTrue(
            ben_franklin_in_response,
            f"Model failed to identify Ben Franklin. Generated: '{generated_text_1}'",
        )

        ### Generation 2 - trims cache
        # create a prompt that replaces the last half of the file content with a copy of the first half
        # this should trigger a cache trim
        prompt = f"""<|im_start|>user
```
{file_content[:int(len(file_content) * 0.5)] + file_content[:int(len(file_content) * 0.5)]}
```
Who is this passage about? Only say the name, and nothing else<end_of_turn>
<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        log_info(
            prefix="test_prompt_caching_trim",
            message=f"Generation 2 number of prompt tokens: {len(prompt_tokens)}",
        )
        generated_text_list_2 = []
        prompt_progress_callback_times_called = 0
        generate(text_accumulator=generated_text_list_2)
        generated_text_2 = "".join(generated_text_list_2)
        # Expect prompt cache ot be intact for the first half of the file_content, so we should get 1
        # intermediate callback this time
        self.assertEqual(prompt_progress_callback_times_called, 3)
        self.assertGreater(len(generated_text_2), 0, "Model failed to generate any text")
        self.assertEqual(generated_text_1, generated_text_2)
