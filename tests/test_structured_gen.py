import unittest
from pathlib import Path

from mlx_engine.generate import create_generator
from .utils import model_helper


class TestStructuredGen(unittest.TestCase):
    def test_structured_gen_with_json_schema(self):
        """Test structured generation with a JSON schema."""
        prompt = "List three colors and their hex codes."
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        model_kit, prompt_tokens = model_helper(model_name, prompt)

        json_schema = """
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

        generator = create_generator(
            model_kit,
            prompt_tokens,
            json_schema=json_schema,
            max_tokens=1024,
        )

        # Collect all generated text
        generated_text = ""
        for generation_result in generator:
            generated_text += generation_result.text
            if generation_result.stop_condition:
                break

        print(f"Generated text:\n{generated_text}")
        # Basic validation that the output looks like JSON
        self.assertTrue(generated_text.strip().startswith("{"))
        self.assertTrue(generated_text.strip().endswith("}"))
        self.assertIn("colors", generated_text)
        self.assertIn("name", generated_text)
        self.assertIn("hex", generated_text)
