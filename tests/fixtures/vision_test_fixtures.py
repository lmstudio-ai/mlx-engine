"""
Shared fixtures and utilities for vision model tests.
This module provides common test patterns, fixtures, and helper functions
to eliminate duplication across vision test files.
"""

import base64
import unittest
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable

import pytest

from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
    is_draft_model_compatible,
)
from tests.shared import model_getter


class VisionTestData:
    """Container for common test data used across vision tests."""

    def __init__(self):
        self.description_prompt = "What is this"
        self.text_only_prompt = "What is a toucan?"
        self.test_data_dir = Path(__file__).parent.parent / "data"
        self.demo_data_dir = Path(__file__).parent.parent.parent / "demo-data"
        self.max_image_size = (1024, 1024)

        # Pre-load and encode test images
        self.toucan_image_b64 = self._encode_image(self.demo_data_dir / "toucan.jpeg")
        self.chameleon_image_b64 = self._encode_image(
            self.demo_data_dir / "chameleon.webp"
        )

    def _encode_image(self, image_path: Path) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _load_model_for_test(
        self, model_path, max_kv_size=4096, trust_remote_code=True
    ):
        """Helper method to load model for testing"""
        from mlx_engine.generate import load_model

        return load_model(
            model_path=model_path,
            max_kv_size=max_kv_size,
            trust_remote_code=trust_remote_code,
        )

    def _tokenize_for_test(self, model_kit, prompt):
        """Helper method to tokenize for testing"""
        from mlx_engine.generate import tokenize

        return tokenize(model_kit, prompt)

    def _create_generator_for_test(self, **kwargs):
        """Helper method to create generator for testing"""
        from mlx_engine.generate import create_generator

        return create_generator(**kwargs)


class VisionTestRunner:
    """Standardized test runner for vision models with common patterns."""

    def __init__(self, test_data: VisionTestData):
        self.test_data = test_data

    def run_vision_test(
        self,
        model_name: str,
        prompt: str,
        text_only: bool = False,
        supplemental_accept_phrases: Optional[List[str]] = None,
        max_tokens: int = 30,
        temp: float = 0.0,
        repetition_penalty: float = 1.01,
        max_kv_size: int = 2048,
        trust_remote_code: bool = True,
        seed: int = 0,
    ) -> str:
        """
        Run a standardized vision test for a model.

        Args:
            model_name: Name/path of the model to test
            prompt: Formatted prompt for the model
            text_only: Whether to test with text only (no images)
            supplemental_accept_phrases: Additional phrases to accept in output
            max_tokens: Maximum tokens to generate
            temp: Temperature for generation
            repetition_penalty: Repetition penalty
            max_kv_size: Maximum KV cache size
            trust_remote_code: Whether to trust remote code
            seed: Random seed

        Returns:
            Generated text
        """
        print(f"Testing model {model_name}")
        model_path = model_getter(model_name=model_name)

        # Load the model
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=max_kv_size,
            trust_remote_code=trust_remote_code,
        )

        # Tokenize the prompt
        prompt_tokens = tokenize(model_kit, prompt)

        # Generate description
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=([self.test_data.toucan_image_b64] if not text_only else None),
            max_image_size=self.test_data.max_image_size,
            seed=seed,
            max_tokens=max_tokens,
            temp=temp,
            repetition_penalty=repetition_penalty,
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print()

        # Verify the output
        assert len(generated_text) > 0, (
            f"Model {model_name} failed to generate any text"
        )

        accept_phrases = ["toucan"]
        if supplemental_accept_phrases:
            accept_phrases += supplemental_accept_phrases

        bird_spotted = any(word in generated_text.lower() for word in accept_phrases)
        assert bird_spotted, (
            f"Model {model_name} failed to generate any of {accept_phrases} in the image. "
            f"Generated: {generated_text.lower()}"
        )

        return generated_text


class CachingTestRunner:
    """Standardized test runner for caching functionality tests."""

    def __init__(self, test_data: VisionTestData):
        self.test_data = test_data

    def run_caching_test(
        self,
        model_name: str,
        story_prompt: str,
        question_prompt: str,
        expected_character: str,
        max_kv_size: int = 4096,
        trust_remote_code: bool = True,
        seed: int = 0,
        temp: float = 0.0,
        max_tokens: int = 1000,
        repetition_penalty: float = 1.01,
    ) -> None:
        """
        Run a standardized caching test for a model.

        Args:
            model_name: Name/path of the model to test
            story_prompt: Prompt to generate a story
            question_prompt: Prompt to ask about the story
            expected_character: Expected character name in the response
            max_kv_size: Maximum KV cache size
            trust_remote_code: Whether to trust remote code
            seed: Random seed
            temp: Temperature for generation
            max_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
        """
        model_path = model_getter(model_name)
        model_kit = load_model(model_path=model_path, max_kv_size=max_kv_size)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> bool:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")
                return True

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=seed,
                temp=temp,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - model creates a long story
        generated_text, num_prompt_processing_callbacks = generate_text(story_prompt)
        assert num_prompt_processing_callbacks == 2, (
            f"Expected 2 callbacks, got {num_prompt_processing_callbacks}"
        )
        assert expected_character in generated_text.lower(), (
            f"Expected '{expected_character}' in generated text"
        )

        # Generation 2 - ask for a detail about the story, should not reprocess
        followup_prompt = story_prompt + generated_text + question_prompt
        num_tokens = len(model_kit.tokenize(followup_prompt))
        # Without caching, prompts > 512 tokens cause multi-batch processing. Ensure prompt meets that condition
        assert num_tokens > 512, (
            f"Prompt should be > 512 tokens to test caching, got {num_tokens}"
        )

        generated_text, num_prompt_processing_callbacks = generate_text(followup_prompt)
        assert num_prompt_processing_callbacks == 2, (
            f"Expected 2 callbacks for cached generation, got {num_prompt_processing_callbacks}"
        )
        assert expected_character.lower() in generated_text.lower(), (
            f"Expected '{expected_character}' in followup response"
        )


class LongPromptCachingTestRunner:
    """Standardized test runner for long prompt caching tests."""

    def __init__(self, test_data: VisionTestData):
        self.test_data = test_data

    def run_long_prompt_caching_test(
        self,
        model_name: str,
        prompt_formatter: Callable[[str], str],
        expected_content: str,
        max_kv_size: int = 4096,
        trust_remote_code: bool = True,
        seed: int = 0,
        temp: float = 0.0,
        max_tokens: int = 1000,
        repetition_penalty: float = 1.01,
        min_tokens: int = 1024,
        expected_batches: int = 4,
    ) -> None:
        """
        Run a standardized long prompt caching test.

        Args:
            model_name: Name/path of the model to test
            prompt_formatter: Function that formats the prompt with file content
            expected_content: Expected content in the response
            max_kv_size: Maximum KV cache size
            trust_remote_code: Whether to trust remote code
            seed: Random seed
            temp: Temperature for generation
            max_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            min_tokens: Minimum tokens the prompt should have
            expected_batches: Expected number of processing batches
        """
        model_path = model_getter(model_name)
        model_kit = load_model(model_path=model_path, max_kv_size=max_kv_size)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> bool:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")
                return True

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=seed,
                temp=temp,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - send model a long excerpt to summarize
        file_path = (
            self.test_data.test_data_dir / "ben_franklin_autobiography_start.txt"
        )
        file_content = file_path.read_text()
        prompt = prompt_formatter(file_content)

        num_tokens = len(model_kit.tokenize(prompt))
        assert num_tokens > min_tokens, (
            f"Prompt should be > {min_tokens} tokens, got {num_tokens}"
        )

        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        assert num_prompt_processing_callbacks == expected_batches, (
            f"Expected {expected_batches} batches, got {num_prompt_processing_callbacks}"
        )
        assert expected_content in generated_text.lower(), (
            f"Expected '{expected_content}' in response"
        )

        # Generation 2 - ask for a detail about the excerpt, should not reprocess
        from textwrap import dedent

        followup_question = dedent("""\
                <end_of_turn>
                <start_of_turn>user
                What was the main characters name?<end_of_turn>
                <start_of_turn>model
                """)
        followup_prompt = prompt + generated_text + followup_question

        print(
            f"Followup prompt length: {len(model_kit.tokenize(followup_prompt))} tokens"
        )
        generated_text, num_prompt_processing_callbacks = generate_text(followup_prompt)
        assert num_prompt_processing_callbacks == 2, (
            f"Expected 2 callbacks for cached followup, got {num_prompt_processing_callbacks}"
        )
        assert expected_content in generated_text.lower(), (
            f"Expected '{expected_content}' in followup response"
        )


# Model configurations for standardized testing
MODEL_CONFIGS = {
    "llama_3_2_vision": {
        "model_name": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
        "vision_prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{description_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "text_prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    },
    "pixtral": {
        "model_name": "mlx-community/pixtral-12b-4bit",
        "vision_prompt": "<s>[INST]{description_prompt}[IMG][/INST]",
        "text_prompt": "<s>[INST]{text_prompt}[/INST]",
        "supplemental_accept_phrases": ["bird"],
    },
    "lfm2_vl": {
        "model_name": "mlx-community/LFM2-VL-450M-4bit",
        "vision_prompt": "<|startoftext|><|im_start|>user\n<image>{description_prompt}<|im_end|>\n <|im_start|>assistant",
        "text_prompt": "<|startoftext|><|im_start|>user\n{text_prompt}<|im_end|>\n <|im_start|>assistant",
    },
    "qwen2_vl": {
        "model_name": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
        "vision_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> {description_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
        "text_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen2_5_vl": {
        "model_name": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "vision_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> {description_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
        "text_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen3_vl": {
        "model_name": "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        "vision_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{description_prompt}<|im_end|>\n<|im_start|>assistant\n",
        "text_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    "qwen3_vl_moe": {
        "model_name": "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit",
        "vision_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{description_prompt}<|im_end|>\n<|im_start|>assistant\n",
        "text_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n",
        "heavy": True,
    },
    "mistral3": {
        "model_name": "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
        "vision_prompt": "<s>[INST]{description_prompt}[IMG][/INST]",
        "text_prompt": "<s>[INST]{text_prompt}[IMG][/INST]",
        "heavy": True,
    },
    "llava": {
        "model_name": "mlx-community/llava-v1.6-mistral-7b-4bit",
        "vision_prompt": "[INST] <image>\n{description_prompt} [/INST]",
        "text_prompt": "[INST] {text_prompt} [/INST]",
        "supplemental_accept_phrases": ["bird"],
    },
    "bunny_llama": {
        "model_name": "mlx-community/Bunny-Llama-3-8B-V-4bit",
        "vision_prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{description_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "text_prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    },
    "nano_llava": {
        "model_name": "mlx-community/nanoLLaVA-1.5-4bit",
        "vision_prompt": "<|im_start|>system\nAnswer the prompt.<|im_end|><|im_start|>user\n<image>\n{description_prompt}<|im_end|><|im_start|>assistant\n\n",
        "text_prompt": "<|im_start|>system\nAnswer the prompt.<|im_end|><|im_start|>user\n{text_prompt}<|im_end|><|im_start|>assistant\n\n",
    },
    "paligemma2": {
        "model_name": "mlx-community/paligemma2-3b-pt-896-4bit",
        "vision_prompt": "<image>{description_prompt}",
        "text_prompt": "{text_prompt}",
        "supplemental_accept_phrases": ["bird"],
        "text_only_unsupported": True,
    },
    "gemma3": {
        "model_name": "mlx-community/gemma-3-4b-it-4bit",
        "vision_prompt": "<bos><start_of_turn>user\n{description_prompt}<start_of_image><end_of_turn>\n<start_of_turn>model\n",
        "text_prompt": "<bos><start_of_turn>user\n{text_prompt}<end_of_turn>\n<start_of_turn>model\n",
        "story_character": "silas",
    },
    "gemma3n": {
        "model_name": "lmstudio-community/gemma-3n-E2B-it-MLX-4bit",
        "vision_prompt": "<bos><start_of_turn>user\n<image_soft_token>{description_prompt}<end_of_turn>\n<start_of_turn>model\n",
        "text_prompt": "<bos><start_of_turn>user\n{text_prompt}<end_of_turn>\n<start_of_turn>model\n",
        "story_character": "silas",
    },
}


# Pytest fixtures
@pytest.fixture
def vision_test_data():
    """Provide VisionTestData instance for tests."""
    return VisionTestData()


@pytest.fixture
def vision_test_runner(vision_test_data):
    """Provide VisionTestRunner instance for tests."""
    return VisionTestRunner(vision_test_data)


@pytest.fixture
def caching_test_runner(vision_test_data):
    """Provide CachingTestRunner instance for tests."""
    return CachingTestRunner(vision_test_data)


@pytest.fixture
def long_prompt_caching_test_runner(vision_test_data):
    """Provide LongPromptCachingTestRunner instance for tests."""
    return LongPromptCachingTestRunner(vision_test_data)


# Base test class for vision tests
class BaseVisionTest(unittest.TestCase):
    """Base class for vision tests with common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods."""
        cls.test_data = VisionTestData()
        cls.vision_runner = VisionTestRunner(cls.test_data)
        cls.caching_runner = CachingTestRunner(cls.test_data)
        cls.long_prompt_runner = LongPromptCachingTestRunner(cls.test_data)

    def run_model_test(self, model_key: str, text_only: bool = False):
        """Run a standardized test for a model configuration."""
        config = MODEL_CONFIGS[model_key]

        prompt = (
            config["text_prompt"].format(text_prompt=self.test_data.text_only_prompt)
            if text_only
            else config["vision_prompt"].format(
                description_prompt=self.test_data.description_prompt
            )
        )

        # Handle models that don't support text-only
        if text_only and config.get("text_only_unsupported"):
            with pytest.raises(
                ValueError,
                match="Using this model without any images attached is not supported yet",
            ):
                self.vision_runner.run_vision_test(
                    config["model_name"],
                    prompt,
                    text_only=True,
                )
            return

        self.vision_runner.run_vision_test(
            config["model_name"],
            prompt,
            text_only=text_only,
            supplemental_accept_phrases=config.get("supplemental_accept_phrases"),
        )

    def run_caching_test_for_model(self, model_key: str):
        """Run a standardized caching test for a model."""
        config = MODEL_CONFIGS[model_key]
        character = config.get("story_character", "clara")

        from textwrap import dedent

        story_prompt = dedent("""\
            <bos><start_of_turn>user
            Tell me a 500-word story<end_of_turn>
            <start_of_turn>model
            """)
        question_prompt = dedent("""\
            <end_of_turn>
            <start_of_turn>user
            What was the main characters name?<end_of_turn>
            <start_of_turn>model
            """)

        self.caching_runner.run_caching_test(
            config["model_name"],
            story_prompt,
            question_prompt,
            character,
        )

    def run_long_prompt_caching_test_for_model(self, model_key: str):
        """Run a standardized long prompt caching test for a model."""
        config = MODEL_CONFIGS[model_key]

        def prompt_formatter(file_content):
            return f"""\
<bos><start_of_turn>user
```
{file_content}
```
Summarize this in one sentence<end_of_turn>
<start_of_turn>model
"""

        self.long_prompt_runner.run_long_prompt_caching_test(
            config["model_name"],
            prompt_formatter,
            "benjamin franklin",
        )
