import json
import unittest

from mlx_engine.generate import create_generator
from .utils import model_load_and_tokenize_prompt


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
            self.model_name, self.prompt, draft_model_name=self.model_name
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
