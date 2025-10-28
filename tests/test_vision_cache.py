import unittest
import base64
from pathlib import Path
from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
)
from tests.shared import model_getter
import pytest

MAX_IMAGE_SIZE = (1024, 1024)


class TestVisionCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
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

    @pytest.mark.heavy
    def test_nonswa_model(self):
        """
        Test that image caching works for models without a SWA cache
        """
        prompt = "<s>[INST][IMG][IMG]In one word each, describe the animal in the images[/INST]\n"
        model_name = "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit"
        model_path = model_getter(model_name=model_name)
        images_b64 = [self.toucan_image_b64, self.chameleon_image_b64]

        # Load the model
        model_kit = load_model(
            model_path=model_path, max_kv_size=4096, trust_remote_code=True
        )

        callback_history = []

        def prompt_callback(x):
            nonlocal callback_history
            callback_history.append(x)
            return True

        def generate_text(prompt):
            nonlocal callback_history
            callback_history = []

            # Tokenize the prompt
            prompt_tokens = tokenize(model_kit, prompt)

            # Generate description
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                prompt_progress_callback=prompt_callback,
                images_b64=images_b64,
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
        self.assertEqual(len(callback_history), 4)  # prompt processing by mlx-lm

        # ask a followup question
        print("--")
        prompt2 = (
            prompt + generated_text + "[INST]what color is each animal's eyes?[/INST]"
        )
        generated_text2 = generate_text(prompt2)

        # prompt processing by cache_wrapper. less work is done since the images are cached
        self.assertEqual(len(callback_history), 2)
        self.assertRegex(generated_text2, "toucan.*dark")
        self.assertRegex(generated_text2, "chameleon.*orange")

        # rewind the cache but swap the images. full preprocessing happens
        images_b64.reverse()
        _ = generate_text(prompt)
        self.assertEqual(len(callback_history), 4)

        # rewind the cache and re-prompt; images are not processed
        _ = generate_text(prompt)
        self.assertEqual(len(callback_history), 2)

        # add an image in the followup; all three images are re-processed
        images_b64.append(self.chameleon_image_b64)
        prompt3 = (
            prompt
            + generated_text
            + "[INST][IMG]do you see two toucans or chameleons?[/INST]"
        )
        generated_text3 = generate_text(prompt3)
        self.assertEqual(len(callback_history), 5)
        self.assertRegex(generated_text3, "chameleon")

    def test_swa_model(self):
        """
        Test that image caching works for models with a SWA cache
        """
        prompt = "<bos><start_of_turn>user\n<image_soft_token><image_soft_token>In one word each, describe the images<end_of_turn>\n<start_of_turn>model\n"
        model_name = "lmstudio-community/gemma-3n-E2B-it-MLX-4bit"
        model_path = model_getter(model_name=model_name)
        images_b64 = [self.toucan_image_b64, self.chameleon_image_b64]

        # Load the model
        model_kit = load_model(
            model_path=model_path, max_kv_size=4096, trust_remote_code=True
        )

        callback_history = []

        def prompt_callback(x):
            nonlocal callback_history
            callback_history.append(x)
            return True

        def generate_text(prompt):
            nonlocal callback_history
            callback_history = []

            # Tokenize the prompt
            prompt_tokens = tokenize(model_kit, prompt)

            # Generate description
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                prompt_progress_callback=prompt_callback,
                images_b64=images_b64,
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
        self.assertEqual(len(callback_history), 4)  # prompt processing by mlx-lm

        # ask a followup question
        print("--")
        prompt2 = (
            prompt
            + generated_text
            + "<end_of_turn>\n<start_of_turn>user\nwhich direction is each animal facing?<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )
        generated_text2 = generate_text(prompt2)

        # prompt processing by cache_wrapper. less work is done since the images are cached
        self.assertEqual(len(callback_history), 2)
        self.assertRegex(generated_text2, "toucan.*left")
        self.assertRegex(generated_text2, "chameleon.*right")

        # swap the images; full preprocessing happens
        images_b64.reverse()
        _ = generate_text(prompt)
        self.assertEqual(len(callback_history), 4)

        # attempt cache rewind; images fully processed since the cache can't be trimmed
        _ = generate_text(prompt)
        self.assertEqual(len(callback_history), 4)

        # add an image in the followup; all three images are re-processed
        images_b64.append(self.chameleon_image_b64)
        prompt3 = (
            prompt
            + generated_text
            + "<start_of_turn>user\n<image_soft_token>how many animals do you count?<end_of_turn>\n<start_of_turn>model"
        )
        generated_text3 = generate_text(prompt3)
        self.assertEqual(len(callback_history), 4)
        self.assertRegex(generated_text3, "chameleon")
