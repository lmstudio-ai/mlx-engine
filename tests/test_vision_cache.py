import unittest
import base64
from pathlib import Path
from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
)
from tests.shared import model_getter


MAX_IMAGE_SIZE = (1024, 1024)


class TestVisionCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.description_prompt = "What is this"
        cls.text_only_prompt = "What is a toucan?"
        cls.test_data_dir = Path(__file__).parent / "data"
        cls.demo_data_dir = Path(__file__).parent.parent / "demo-data"

        # Read and encode test images
        cls.toucan_path = Path(__file__).parent.parent / "demo-data" / "toucan.jpeg"
        with open(cls.toucan_path, "rb") as image_file:
            cls.toucan_image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        cls.chameleon_image_path = (
            Path(__file__).parent.parent / "demo-data" / "chameleon.webp"
        )
        with open(cls.chameleon_image_path, "rb") as chameleon_image_file:
            cls.chameleon_image_b64 = base64.b64encode(
                chameleon_image_file.read()
            ).decode("utf-8")

    ### MODEL-SPECIFIC TESTS ###
    def test_gemma3n(self):
        """Test LFM2-VL 450M model"""
        prompt = "<bos><start_of_turn>user\n<image_soft_token><image_soft_token>In one word each, describe the images<end_of_turn>\n<start_of_turn>model\n"
        model_name = "lmstudio-community/gemma-3n-E2B-it-MLX-4bit"
        print(f"Testing model {model_name}")
        model_path = model_getter(model_name=model_name)

        # Load the model
        model_kit = load_model(
            model_path=model_path, max_kv_size=2048, trust_remote_code=True
        )

        def generate_text(prompt):
            # Tokenize the prompt
            prompt_tokens = tokenize(model_kit, prompt)

            # Generate description
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=[self.toucan_image_b64, self.chameleon_image_b64],
                max_image_size=MAX_IMAGE_SIZE,
                seed=0,
                max_tokens=100,
                temp=0.0,
                repetition_penalty=1.01,  # enable the logits processor code path
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print()
            return generated_text

        generated_text = generate_text(prompt)
        print("--")
        prompt = (
            prompt
            + generated_text
            + "<end_of_turn>\n<start_of_turn>user\nwhich direction is each animal facing?<end_of_turn>\n<start_of_turn>model\n"
        )
        generated_text = generate_text(prompt)
