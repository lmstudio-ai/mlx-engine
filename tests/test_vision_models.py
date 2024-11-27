import unittest
import base64
from pathlib import Path
from mlx_engine.generate import load_model, tokenize, create_generator
import sys
import subprocess
import os

class TestVisionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.toucan_path = Path("demo-data/toucan.jpeg")
        cls.model_path_prefix = Path("~/.cache/lm-studio/models/mlx-community").expanduser().resolve()
        cls.description_prompt = "Describe what you see in great detail"
        
        # Read and encode the test image
        with open(cls.toucan_path, "rb") as image_file:
            cls.image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

    def model_helper(self, model_name: str, prompt: str):
        """Helper method to test a single vision model"""
        print(f"Testing model {model_name}")

        model_path = self.model_path_prefix / model_name
        
        # Load the model
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=2048,
            trust_remote_code=True
        )
        
        # Tokenize the prompt
        prompt_tokens = tokenize(model_kit, prompt)
        
        # Generate description
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=[self.image_b64],
            seed=0,
            max_tokens=20,
            temp=0.0,
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print()
        
        # Verify the output
        self.assertGreater(len(generated_text), 0, f"Model {model_name} failed to generate any text")
        bird_spotted = any(word in generated_text.lower() for word in ["bird", "toucan", "quetzal"])
        self.assertTrue(bird_spotted,
                       f"Model {model_name} failed to identify either a bird in the image")
        
        return generated_text

    def test_llama_vision_instruct(self):
        """Test Llama 3.2 11B Vision Instruct model"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.description_prompt}<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        self.model_helper("Llama-3.2-11B-Vision-Instruct-4bit", prompt)

    def test_pixtral(self):
        """Test Pixtral 12B model"""
        prompt = f"<s>[INST]{self.description_prompt}[IMG][/INST]"
        self.model_helper("pixtral-12b-4bit", prompt)

    def test_qwen2(self):
        """Test Qwen2 VL 7B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> {self.description_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        self.model_helper("Qwen2-VL-7B-Instruct-4bit", prompt)

    def test_florence(self):
        """Test Florence 2 Large model"""
        prompt = self.description_prompt
        self.model_helper("Florence-2-large-ft-bf16", prompt)

    def test_molmo(self):
        """Test Molmo 7B model"""
        prompt = self.description_prompt
        self.model_helper("Molmo-7B-D-0924-4bit", prompt)

    def test_llava(self):
        """Test LLaVA v1.6 Mistral 7B model"""
        prompt = f"<image> {self.description_prompt}"
        self.model_helper("llava-v1.6-mistral-7b-4bit", prompt)

    def test_bunny_llama(self):
        """Test Bunny Llama 3 8B V model"""
        prompt = f"<image> {self.description_prompt}"
        self.model_helper("Bunny-Llama-3-8B-V-4bit", prompt)

    def test_nano_llava(self):
        """Test Nano LLaVA 1.5 4B model"""
        prompt = f"<|im_start|>system\nAnswer the prompt.<|im_end|><|im_start|>user\n<image>\n{self.description_prompt}<|im_end|><|im_start|>assistant\n\n"
        self.model_helper("nanoLLaVA-1.5-4bit", prompt)

def run_single_test(test_name):
    """Run a single test in isolation"""
    suite = unittest.TestSuite()
    suite.addTest(TestVisionModels(test_name))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    # If a test name is provided as an argument, run just that test
    if len(sys.argv) > 1:
        run_single_test(sys.argv[1])
        sys.exit(0)

    # List of all test methods
    test_methods = [
        'test_llama_vision_instruct',
        'test_pixtral',
        'test_qwen2',
        'test_florence',
        'test_molmo',
        'test_llava',
        'test_bunny_llama',
        'test_nano_llava',
    ]
    
    # Get the current script path
    script_path = os.path.abspath(__file__)
    
    # Run each test in a separate Python process, to control memory usage
    for test_name in test_methods:
        print(f"\nStarting process for {test_name}")
        
        # Launch a new Python interpreter process for this test
        result = subprocess.run(
            [sys.executable, script_path, test_name],
            capture_output=True,
            text=True
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Check if the test passed
        if result.returncode != 0:
            print(f"Test {test_name} failed!")
            sys.exit(1)
        
        # Force cleanup
        print(f"Completed {test_name}\n")
