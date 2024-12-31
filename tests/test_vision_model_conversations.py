import unittest
import base64
from pathlib import Path
import sys
import subprocess
import os

from mlx_engine.generate import load_model, tokenize, create_generator


class TestVisionConversations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        # Get the project root directory (parent of the tests directory)
        cls.project_root = Path(__file__).parent.parent
        
        # Setup paths and test images relative to project root
        cls.toucan_path = cls.project_root / "demo-data" / "toucan.jpeg"
        cls.chameleon_path = cls.project_root / "demo-data" / "chameleon.webp"
        cls.model_path_prefix = Path("~/.cache/lm-studio/models").expanduser().resolve()

        # Common prompts for conversation flows
        cls.initial_prompt = "What animal do you see in this image?"
        cls.followup_prompt = "Which one would make a better pet and why?"
        cls.detail_prompt = "Describe their colors in detail."

        # Read and encode test images
        with open(cls.toucan_path, "rb") as image_file:
            cls.toucan_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        with open(cls.chameleon_path, "rb") as image_file:
            cls.chameleon_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    def model_helper(self, model_name: str, prompts: list, images_list: list = None):
        """Helper method to test a vision model with a conversation"""
        print(f"\nTesting model {model_name}")
        model_path = self.model_path_prefix / model_name

        # Check if model exists, if not prompt user to download
        if not model_path.exists():
            print(f"\nModel {model_name} not found at {model_path}")

            def greenify(text):
                return f"\033[92m{text}\033[0m"

            response = input(
                f"Would you like to download the model {greenify(model_name)}? (y/N): "
            )
            if response.lower() == "y":
                print(f"Downloading model with command: lms get {model_name}")
                subprocess.run(["lms", "get", model_name], check=True)
            else:
                print(f"Model {model_name} not found")
                sys.exit(1)

        # Load the model
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=2048,
            trust_remote_code=True
        )

        conversation_history = ""
        responses = []

        for i, prompt in enumerate(prompts):
            print(f"\nTurn {i+1}:")
            current_prompt = conversation_history + prompt
            prompt_tokens = tokenize(model_kit, current_prompt)

            # Get current images for this turn, if any
            current_images = images_list[i] if images_list and i < len(images_list) else None

            # Generate response
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=current_images,
                seed=0,
                max_tokens=100,
                temp=0.0,
                repetition_penalty=1.01,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print()

            # Update conversation history and store response
            conversation_history = current_prompt + generated_text
            responses.append(generated_text)

            # Basic validation for each turn
            self.assertGreater(
                len(generated_text),
                0,
                f"Model {model_name} failed to generate text in turn {i+1}"
            )

        return responses

    def test_llava_conversation(self):
        """Test LLaVA with a multi-turn conversation including multiple images"""
        prompts = [
            f"[INST] <image>\n{self.initial_prompt} [/INST]",
            f"[INST] Now look at this other image: <image>\nCompare these two animals. [/INST]",
            f"[INST] {self.followup_prompt} [/INST]",
        ]
        images = [
            [self.toucan_b64],
            [self.chameleon_b64],
            None
        ]
        responses = self.model_helper(
            "mlx-community/llava-v1.6-mistral-7b-4bit",
            prompts,
            images
        )
        
        # Verify responses
        self.assertTrue(
            any(word in responses[0].lower() for word in ["toucan", "bird"]),
            "Failed to identify toucan in first image"
        )
        self.assertTrue(
            any(word in responses[1].lower() for word in ["chameleon", "reptile", "lizard"]),
            "Failed to identify chameleon in second image"
        )
        self.assertTrue(
            any(word in responses[2].lower() for word in ["pet", "domestic", "house"]),
            "Failed to discuss pet suitability"
        )

    def test_qwen_conversation(self):
        """Test Qwen2 with a multi-turn conversation about multiple images"""
        prompts = [
            f"<|im_start|>user\n<image>\n{self.initial_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
            f"<|im_start|>user\nNow look at this: <image>\n{self.detail_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
            f"<|im_start|>user\n{self.followup_prompt}<|im_end|>\n<|im_start|>assistant\n"
        ]
        images = [
            [self.toucan_b64],
            [self.chameleon_b64],
            None
        ]
        responses = self.model_helper(
            "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            prompts,
            images
        )
        
        # Verify responses
        self.assertTrue(
            any(word in responses[0].lower() for word in ["toucan", "bird"]),
            "Failed to identify toucan"
        )
        self.assertTrue(
            any(word in responses[1].lower() for word in ["color", "pattern", "green", "scales"]),
            "Failed to describe colors"
        )
        self.assertTrue(
            any(word in responses[2].lower() for word in ["better", "recommend", "suitable"]),
            "Failed to make pet recommendation"
        )

    def test_pixtral_conversation(self):
        """Test Pixtral with a multi-turn conversation"""
        prompts = [
            f"<s>[INST]{self.initial_prompt}[IMG][/INST]",
            f"<s>[INST]And what about this one? [IMG][/INST]",
            f"<s>[INST]{self.followup_prompt}[/INST]"
        ]
        images = [
            [self.toucan_b64],
            [self.chameleon_b64],
            None
        ]
        responses = self.model_helper(
            "mlx-community/pixtral-12b-4bit",
            prompts,
            images
        )
        
        # Verify responses
        self.assertTrue(
            any(word in responses[0].lower() for word in ["toucan", "bird"]),
            "Failed to identify first animal"
        )
        self.assertTrue(
            any(word in responses[1].lower() for word in ["chameleon", "reptile", "lizard"]),
            "Failed to identify second animal"
        )
        self.assertTrue(
            any(word in ' '.join(responses).lower() for word in ["wild", "exotic", "pet"]),
            "Failed to compare animals"
        )


def run_single_test(test_name):
    """Run a single test in isolation"""
    suite = unittest.TestSuite()
    suite.addTest(TestVisionConversations(test_name))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    # If a test name is provided as an argument, run just that test
    if len(sys.argv) > 1:
        run_single_test(sys.argv[1])
        sys.exit(0)

    # List of all test methods
    test_methods = [
        "test_llava_conversation",
        "test_qwen_conversation",
        "test_pixtral_conversation",
    ]

    # Get the current script path
    script_path = os.path.abspath(__file__)

    # Run each test in a separate Python process
    for test_name in test_methods:
        print(f"\nStarting process for {test_name}")
        
        # Launch new Python interpreter process for this test
        result = subprocess.run(
            [sys.executable, script_path, test_name],
            capture_output=True,
            text=True
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Check if test passed
        if result.returncode != 0:
            print(f"Test {test_name} failed!")
            sys.exit(1)

        # Force cleanup
        print(f"Completed {test_name}\n")
