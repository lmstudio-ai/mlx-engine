import json
import pytest
import unittest
from pathlib import Path
import logging

from tests.shared import (
    model_getter,
    model_load_and_tokenize_prompt,
    RecordingReporter,
)
from mlx_engine.generate import (
    load_model,
    load_draft_model,
    is_draft_model_compatible,
    unload_draft_model,
    tokenize,
    create_generator,
    unload,
)

logger = logging.getLogger(__name__)


class TestTextModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.test_data_dir = Path(__file__).parent / "data"

    def test_repetition_penalty_applies(self):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)
        self.addCleanup(lambda *_: unload(model_kit))
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
                repetition_penalty=3,  # this will make it so that model shouldn't repeat. If set to 0, it will
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
        self.assertNotIn(
            "The quick brown fox jumped over the lazy dog.", generated_text
        )

    def test_prompt_caching_happy_path_qwen2_5(self):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=20000)
        self.addCleanup(lambda *_: unload(model_kit))
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        prompt = f"""<|im_start|>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        reporter = RecordingReporter()

        def generate() -> None:
            nonlocal generated_text
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                max_tokens=10,
                temp=0.0,
                prompt_progress_reporter=reporter,
            ):
                print(result.text, end="", flush=True)
                generated_text += result.text
                if result.stop_condition:
                    break
            print("\n", flush=True)

        def reset_state() -> None:
            nonlocal generated_text
            generated_text = ""
            reporter.events.clear()

        ### Generation 1
        generate()
        self.assertEqual(len(reporter.events), 5)
        # First generation should have no cached tokens
        begin_event = reporter.events[0]
        self.assertEqual(begin_event["type"], "begin")
        # TODO: Implement proper cached_tokens tracking in batched model kit
        # Currently hardcoded to 0 in BatchedMlxLmReporterAdapter
        # self.assertEqual(begin_event["cached_tokens"], 0)
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        ben_franklin_in_response = "Benjamin Franklin" in generated_text
        self.assertTrue(
            ben_franklin_in_response,
            f"Model failed to identify Ben Franklin. Generated: '{generated_text}'",
        )

        prompt += generated_text
        prompt += """<|im_end|>
<|im_start|>user
repeat<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        reset_state()

        ### Generation 2
        generate()
        # Expect prompt cache to be intact, so we should only get begin and finish callbacks
        self.assertEqual(len(reporter.events), 2)
        # Second generation should have cached tokens from first generation
        begin_event = reporter.events[0]
        self.assertEqual(begin_event["type"], "begin")
        # self.assertGreater(begin_event["cached_tokens"], 0)
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        ben_franklin_in_response = "Benjamin Franklin" in generated_text
        self.assertTrue(
            ben_franklin_in_response,
            f"Model failed to identify Ben Franklin. Generated: '{generated_text}'",
        )

    def test_prompt_caching_trim_qwen2_5(self):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=20000)
        self.addCleanup(lambda *_: unload(model_kit))
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        prompt = f"""<|im_start|>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<end_of_turn>
<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        logger.info(
            f"Generation 1 number of prompt tokens: {len(prompt_tokens)}",
        )
        generated_text_list_1 = []
        reporter = RecordingReporter()

        # accumulating to list allows pass by reference
        def generate(text_accumulator: list) -> None:
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                max_tokens=10,
                temp=0.0,
                prompt_progress_reporter=reporter,
            ):
                print(result.text, end="", flush=True)
                text_accumulator.append(result.text)
                if result.stop_condition:
                    break
            print("\n", flush=True)

        ### Generation 1 - fills cache
        generate(text_accumulator=generated_text_list_1)
        generated_text_1 = "".join(generated_text_list_1)
        self.assertEqual(len(reporter.events), 5)
        # First generation should have no cached tokens
        begin_event = reporter.events[0]
        self.assertEqual(begin_event["type"], "begin")
        # TODO: Implement proper cached_tokens tracking in batched model kit
        # Currently hardcoded to 0 in BatchedMlxLmReporterAdapter
        # self.assertEqual(begin_event["cached_tokens"], 0)
        self.assertGreater(
            len(generated_text_1), 0, "Model failed to generate any text"
        )
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
{file_content[: int(len(file_content) * 0.5)] + file_content[: int(len(file_content) * 0.5)]}
```
Who is this passage about? Only say the name, and nothing else<end_of_turn>
<|im_end|>
<|im_start|>assistant
"""
        prompt_tokens = tokenize(model_kit, prompt)
        logger.info(
            f"Generation 2 number of prompt tokens: {len(prompt_tokens)}",
        )
        generated_text_list_2 = []
        reporter.events.clear()
        generate(text_accumulator=generated_text_list_2)
        generated_text_2 = "".join(generated_text_list_2)
        # Expect prompt cache to be intact for the first half of the file_content, so we should get 1
        # intermediate update callback this time (begin + 1x update + finish = 4)
        self.assertEqual(len(reporter.events), 3)
        # Second generation should have some cached tokens (partial cache hit after trim)
        begin_event = reporter.events[0]
        self.assertEqual(begin_event["type"], "begin")
        # self.assertGreater(begin_event["cached_tokens"], 0)
        self.assertGreater(
            len(generated_text_2), 0, "Model failed to generate any text"
        )
        self.assertEqual(generated_text_1, generated_text_2)

    def test_kv_cache_quantization_lfm2(self):
        """Test KV Cache Quantization with a Hybrid model"""
        model_path = model_getter("lmstudio-community/LFM2-350M-MLX-8bit")
        model_kit = load_model(
            model_path=model_path, max_kv_size=4096, kv_bits=8, kv_group_size=64
        )
        self.addCleanup(lambda *_: unload(model_kit))
        prompt_template = """<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user{user_text}
<|im_start|>assistant
"""
        prompt = prompt_template.format(
            user_text="Write a function to check is_palindrome in python"
        )
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            seed=0,
            temp=0.0,
            max_tokens=100,
        ):
            print(result.text, end="", flush=True)
            generated_text += result.text
            if result.stop_condition:
                break
        print("\n", flush=True)
        # Just check that output is reasonable. Should include the function name we requested
        self.assertTrue("def is_palindrome(" in generated_text)


class TestStructuredGen(unittest.TestCase):
    def setUp(self):
        self.prompt = "List three colors and their hex codes."
        self.model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        self.json_schema = """
        {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"}
                        },
                        "required": ["name", "hex"]
                    }
                }
            },
            "required": ["colors"]
        }
        """

    def test_structured_gen_with_json_schema(self):
        model_kit, prompt_tokens = model_load_and_tokenize_prompt(
            self.model_name, self.prompt
        )
        self.addCleanup(lambda *_: unload(model_kit))

        generator = create_generator(
            model_kit,
            prompt_tokens,
            json_schema=self.json_schema,
            max_tokens=1024,
            seed=0,
        )

        # Collect all generated text
        generated_text = ""
        for generation_result in generator:
            generated_text += generation_result.text
            if generation_result.stop_condition:
                break

        # Basic validation that the output looks like JSON
        print(f"Generated text:\n{generated_text}")
        self.assertTrue(generated_text.strip().startswith("{"))
        self.assertTrue(generated_text.strip().endswith("}"))
        self.assertIn("colors", generated_text)
        self.assertIn("name", generated_text)
        self.assertIn("hex", generated_text)

        # throw if not valid JSON
        json.loads(generated_text)

    def test_structured_gen_with_json_schema_speculative_decoding(self):
        # Uses same model for main and draft, not a speed test
        model_kit, prompt_tokens = model_load_and_tokenize_prompt(
            self.model_name,
            self.prompt,
            draft_model_name=self.model_name,
            max_num_seqs=1,
        )

        generator = create_generator(
            model_kit,
            prompt_tokens,
            json_schema=self.json_schema,
            max_tokens=1024,
            seed=0,
        )

        generated_text = ""
        for generation_result in generator:
            generated_text += generation_result.text
            if generation_result.stop_condition:
                break

        print(f"Generated text:\n{generated_text}")
        self.assertTrue(generated_text.strip().startswith("{"))
        self.assertTrue(generated_text.strip().endswith("}"))
        self.assertIn("colors", generated_text)
        self.assertIn("name", generated_text)
        self.assertIn("hex", generated_text)

        # throw if not valid JSON
        json.loads(generated_text)

    @pytest.mark.heavy
    def test_structured_gen_rnj_1(self):
        """Test Structured Generation with RNJ-1 model"""
        model_path = model_getter("lmstudio-community/rnj-1-instruct-MLX-4bit")
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=4096,
        )
        self.addCleanup(lambda *_: unload(model_kit))
        prompt_template = """<|start_header_id|>system<|end_header_id|>
You are rnj-1, a foundation model trained by Essential AI.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        prompt = prompt_template.format(user_text=self.prompt)
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            seed=0,
            temp=0.0,
            max_tokens=100,
            json_schema=self.json_schema,
        ):
            print(result.text, end="", flush=True)
            generated_text += result.text
            if result.stop_condition:
                break
        print("\n", flush=True)

        # This is what the structured output would force without EOT token correction
        assert "<|end_of_text|>" not in generated_text
        # throw if not valid JSON
        json.loads(generated_text)


class TestSpeculativeDecoding(unittest.TestCase):
    def test_is_draft_model_compatible_true_vocab_only_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, vocab_only=True, max_num_seqs=1)
        self.assertTrue(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_true_full_model_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_num_seqs=1)
        self.assertTrue(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_false_vocab_only_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, vocab_only=True, max_num_seqs=1)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_is_draft_model_compatible_false_full_model_load(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_num_seqs=1)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )

    def test_load_draft_model_success(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None, max_num_seqs=1)
        load_draft_model(model_kit=model_kit, path=draft_model_path)
        self.assertIsNotNone(model_kit.draft_model)

    def test_load_draft_model_invalid_model(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter("mlx-community/Llama-3.2-1B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=None, max_num_seqs=1)
        with self.assertRaises(ValueError):
            load_draft_model(model_kit=model_kit, path=draft_model_path)

    def test_unload_draft_model_idempotent_none_loaded(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=None, max_num_seqs=1)
        unload_draft_model(model_kit=model_kit)

    def test_unload_draft_model_success(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None, max_num_seqs=1)
        model_kit.load_draft_model(path=draft_model_path)
        unload_draft_model(model_kit=model_kit)
        self.assertIsNone(model_kit.draft_model)

    def test_basic_generation(self):
        model_path = model_getter("mlx-community/Qwen2.5-3B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path, max_kv_size=None, max_num_seqs=1)
        load_draft_model(model_kit=model_kit, path=draft_model_path)
        prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenize(model_kit, prompt)
        generated_text = ""
        from_draft_count = 0
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            seed=0,
            max_tokens=10,
            temp=0.0,
        ):
            for token in result.tokens:
                if token.from_draft:
                    from_draft_count += 1
            generated_text += result.text
            if result.stop_condition:
                break

        # Verify the output
        self.assertGreater(len(generated_text), 0, "Model failed to generate any text")
        paris_in_response = "paris" in generated_text.lower()
        self.assertTrue(
            paris_in_response,
            "Model failed to respond correctly",
        )
        self.assertGreaterEqual(
            from_draft_count,
            3,
            "Less draft tokens accepted than expected",
        )
