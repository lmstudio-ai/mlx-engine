"""
Refactored vision model tests using shared fixtures and utilities.
This module eliminates duplication by using standardized test patterns.
"""

import unittest
import pytest
from textwrap import dedent

from tests.fixtures.vision_test_fixtures import (
    BaseVisionTest,
    MODEL_CONFIGS,
)
from tests.shared import model_getter
from mlx_engine.generate import is_draft_model_compatible


class TestVisionModels(BaseVisionTest):
    """Test suite for vision models using shared fixtures and utilities."""

    ### MODEL-SPECIFIC TESTS ###

    def test_llama_3_2_vision_instruct(self):
        """Test Llama 3.2 11B Vision Instruct model"""
        self.run_model_test("llama_3_2_vision")

    def test_llama_3_2_vision_instruct_text_only(self):
        """Test Llama 3.2 11B Vision Instruct model with only text"""
        self.run_model_test("llama_3_2_vision", text_only=True)

    def test_pixtral_vision(self):
        """Test Pixtral 12B model"""
        self.run_model_test("pixtral")

    def test_pixtral_text_only(self):
        """Test Pixtral 12B model with only text"""
        self.run_model_test("pixtral", text_only=True)

    def test_lfm2_vl_vision(self):
        """Test LFM2-VL 450M model"""
        self.run_model_test("lfm2_vl")

    def test_lfm2_vl_text_only(self):
        """Test LFM2-VL 450M model"""
        self.run_model_test("lfm2_vl", text_only=True)

    @pytest.mark.heavy
    def test_mistral3_vision(self):
        """Test Mistral3 model"""
        self.run_model_test("mistral3")

    @pytest.mark.heavy
    def test_mistral3_text_only(self):
        """Test Mistral3 model with only text"""
        self.run_model_test("mistral3", text_only=True)

    def test_qwen2_vision(self):
        """Test Qwen2 VL 7B Instruct model"""
        self.run_model_test("qwen2_vl")

    def test_qwen2_text_only(self):
        """Test Qwen2 VL 7B Instruct model with only text"""
        self.run_model_test("qwen2_vl", text_only=True)

    def test_qwen2_5_vision(self):
        """Test Qwen2.5 VL 7B Instruct model"""
        self.run_model_test("qwen2_5_vl")

    def test_qwen2_5_text_only(self):
        """Test Qwen2.5 VL 7B Instruct model with only text"""
        self.run_model_test("qwen2_5_vl", text_only=True)

    def test_qwen3_vl_vision(self):
        """Test Qwen3-VL 4B Instruct model"""
        self.run_model_test("qwen3_vl")

    def test_qwen3_vl_text_only(self):
        """Test Qwen3-VL 4B Instruct model with only text"""
        self.run_model_test("qwen3_vl", text_only=True)

    @pytest.mark.heavy
    def test_qwen3_vl_moe_vision(self):
        """Test Qwen3-VL 30B-A3B Instruct model"""
        self.run_model_test("qwen3_vl_moe")

    @pytest.mark.heavy
    def test_qwen3_vl_moe_text_only(self):
        """Test Qwen3-VL 30B-A3B Instruct model with only text"""
        self.run_model_test("qwen3_vl_moe", text_only=True)

    def test_llava_vision(self):
        """Test LLaVA v1.6 Mistral 7B model"""
        self.run_model_test("llava")

    def test_llava_text_only(self):
        """Test LLaVA v1.6 Mistral 7B model with only text"""
        self.run_model_test("llava", text_only=True)

    def test_bunny_llama_vision(self):
        """Test Bunny Llama 3 8B V model"""
        self.run_model_test("bunny_llama")

    def test_bunny_llama_text_only(self):
        """Test Bunny Llama 3 8B V model with only text"""
        self.run_model_test("bunny_llama", text_only=True)

    def test_nano_llava_vision(self):
        """Test Nano LLaVA 1.5 4B model"""
        self.run_model_test("nano_llava")

    def test_nano_llava_text_only(self):
        """Test Nano LLaVA 1.5 4B model with only text"""
        self.run_model_test("nano_llava", text_only=True)

    def test_paligemma2_vision(self):
        """Test Paligemma 2 model"""
        self.run_model_test("paligemma2")

    def test_paligemma2_text_only(self):
        """Test Paligemma 2 model with only text"""
        self.run_model_test("paligemma2", text_only=True)

    def test_gemma3_vision(self):
        """Test Gemma 3 model"""
        self.run_model_test("gemma3")

    def test_gemma3_text_only_short(self):
        """Test Gemma 3 model"""
        self.run_model_test("gemma3", text_only=True)

    def test_gemma3n_vision(self):
        """Test gemma 3n model"""
        self.run_model_test("gemma3n")

    def test_gemma3n_text_only(self):
        """Test gemma 3n model text only"""
        self.run_model_test("gemma3n", text_only=True)

    ### CACHING TESTS ###

    @pytest.mark.heavy
    def test_mistral3_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        self.run_caching_test_for_model("mistral3")

    def test_gemma3_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        self.run_caching_test_for_model("gemma3")

    def test_gemma3n_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        self.run_caching_test_for_model("gemma3n")

    def test_gemma3_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        self.run_long_prompt_caching_test_for_model("gemma3")

    def test_gemma3n_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        self.run_long_prompt_caching_test_for_model("gemma3n")

    def test_gemma3n_vision_long_prompt_progress_reported(self):
        """Ensure progress is reported during prompt processing with a vision prompt"""
        config = MODEL_CONFIGS["gemma3n"]

        file_path = (
            self.test_data.test_data_dir / "ben_franklin_autobiography_start.txt"
        )
        file_content = file_path.read_text()

        prompt = f"""\
<bos><start_of_turn>user
<image_soft_token>
```
{file_content}
```
Summarize this in one sentence<end_of_turn>
<start_of_turn>model
"""

        model_path = model_getter(config["model_name"])
        model_kit = self.vision_runner.test_data._load_model_for_test(
            model_path, max_kv_size=4096
        )

        prompt_tokens = self.vision_runner.test_data._tokenize_for_test(
            model_kit, prompt
        )
        progress_values = []

        def progress_callback(progress: float) -> bool:
            progress_values.append(progress)
            print(f"Prompt processing progress: {progress}")
            return True

        generated_text = ""
        for result in self.vision_runner.test_data._create_generator_for_test(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=[self.test_data.toucan_image_b64],
            max_image_size=self.test_data.max_image_size,
            seed=0,
            temp=0.0,
            max_tokens=1,  # We only care about pre-fill in this test
            repetition_penalty=1.01,
            prompt_progress_callback=progress_callback,
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print("\n", flush=True)
        print(progress_values)

        assert len(progress_values) > 0, "No progress values reported"
        for i in range(len(progress_values) - 1):
            assert progress_values[i + 1] > progress_values[i], (
                f"Progress should increase: {progress_values[i]} -> {progress_values[i + 1]}"
            )

    ### MULTI-IMAGE TESTS ###

    def test_qwen2_5_images_across_messages(self):
        """Test Qwen2.5 with multiple images across messages"""
        config = MODEL_CONFIGS["qwen2_5_vl"]

        model_path = model_getter(config["model_name"])
        model_kit = self.vision_runner.test_data._load_model_for_test(
            model_path, max_kv_size=4096
        )

        def generate_text(prompt, images_b64):
            prompt_tokens = self.vision_runner.test_data._tokenize_for_test(
                model_kit, prompt
            )
            generated_text = ""
            for result in self.vision_runner.test_data._create_generator_for_test(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                max_image_size=self.test_data.max_image_size,
                seed=0,
                temp=0.0,
                max_tokens=50,
                repetition_penalty=1.01,  # to enable this code path
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text

        # Test case 1: Single image
        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is this? In one word.<|im_end|>
<|im_start|>assistant
"""
        images_b64 = [self.test_data.toucan_image_b64]
        generated_text = generate_text(prompt, images_b64)
        # The logits for "Bird" and "T" are incredibly close to each other for this generation.
        # Therefore, accept either to reduce flakiness, as both "toucan" and "bird" are acceptable.
        acceptable_words = ["toucan", "bird"]
        is_word_accepted = any(
            word in generated_text.lower() for word in acceptable_words
        )
        assert is_word_accepted, (
            f"Expected one of {acceptable_words} but got {generated_text.lower()}"
        )

        # Test case 2: Second image added in continued conversation
        prompt += """Toucan.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>In a few words, describe all of the images you've seen so far<|im_end|>
<|im_start|>assistant
"""
        images_b64 = [
            self.test_data.toucan_image_b64,
            self.test_data.chameleon_image_b64,
        ]
        generated_text = generate_text(prompt, images_b64)
        assert "toucan" in generated_text.lower(), "Expected 'toucan' in response"
        assert "chameleon" in generated_text.lower(), "Expected 'chameleon' in response"

    ### SKIPPED TESTS ###

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_florence_vision(self):
        """Test Florence 2 Large model"""
        config = MODEL_CONFIGS.get(
            "florence",
            {
                "model_name": "mlx-community/Florence-2-base-ft-4bit",
                "vision_prompt": "{description_prompt}",
                "text_prompt": "{text_prompt}",
            },
        )
        prompt = config["vision_prompt"].format(
            description_prompt=self.test_data.description_prompt
        )
        self.vision_runner.run_vision_test(config["model_name"], prompt)

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_florence_text_only(self):
        """Test Florence 2 Large model with only text"""
        config = MODEL_CONFIGS.get(
            "florence",
            {
                "model_name": "mlx-community/Florence-2-base-ft-4bit",
                "vision_prompt": "{description_prompt}",
                "text_prompt": "{text_prompt}",
            },
        )
        prompt = config["text_prompt"].format(
            text_prompt=self.test_data.text_only_prompt
        )
        with pytest.raises(
            ValueError,
            match="Using this model without any images attached is not supported yet",
        ):
            self.vision_runner.run_vision_test(
                config["model_name"], prompt, text_only=True
            )

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_molmo_vision(self):
        """Test Molmo 7B model"""
        config = MODEL_CONFIGS.get(
            "molmo",
            {
                "model_name": "mlx-community/Molmo-7B-D-0924-4bit",
                "vision_prompt": "{description_prompt}",
                "text_prompt": "{text_prompt}",
            },
        )
        prompt = config["vision_prompt"].format(
            description_prompt=self.test_data.description_prompt
        )
        self.vision_runner.run_vision_test(config["model_name"], prompt)

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_molmo_text_only(self):
        """Test Molmo 7B model with only text"""
        config = MODEL_CONFIGS.get(
            "molmo",
            {
                "model_name": "mlx-community/Molmo-7B-D-0924-4bit",
                "vision_prompt": "{description_prompt}",
                "text_prompt": "{text_prompt}",
            },
        )
        prompt = config["text_prompt"].format(
            text_prompt=self.test_data.text_only_prompt
        )
        self.vision_runner.run_vision_test(config["model_name"], prompt, text_only=True)

    ### NON-MODEL-SPECIFIC TESTS ###

    def test_draft_model_not_compatible_vision(self):
        model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = self.vision_runner.test_data._load_model_for_test(model_path)
        assert not is_draft_model_compatible(
            model_kit=model_kit, path=draft_model_path
        ), "Vision models should not be compatible with draft models"


"""
To find the correct prompt format for new models, run this command for your model in the terminal and check the prompt dump:
python -m mlx_vlm.generate --model ~/.cache/lm-studio/models/mlx-community/MODEL-NAME --max-tokens 100 --temp 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg --prompt "What do you see?"
"""
