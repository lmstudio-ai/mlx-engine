import unittest
import base64
from pathlib import Path
import pytest
from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
    is_draft_model_compatible,
)
from tests.shared import model_getter
from textwrap import dedent


class TestVisionModels(unittest.TestCase):
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

    def toucan_test_runner(
        self,
        model_name: str,
        prompt: str,
        text_only=False,
        supplemental_accept_phrases=None,
    ):
        """Helper method to test a single vision model"""
        print(f"Testing model {model_name}")
        model_path = model_getter(model_name=model_name)

        # Load the model
        model_kit = load_model(
            model_path=model_path, max_kv_size=2048, trust_remote_code=True
        )

        # Tokenize the prompt
        prompt_tokens = tokenize(model_kit, prompt)

        # Generate description
        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=([self.toucan_image_b64] if not text_only else None),
            seed=0,
            max_tokens=30,
            temp=0.0,
            repetition_penalty=1.01,  # enable the logits processor code path
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print()

        # Verify the output
        self.assertGreater(
            len(generated_text), 0, f"Model {model_name} failed to generate any text"
        )
        accept_phrases = ["toucan"]
        if supplemental_accept_phrases:
            accept_phrases += supplemental_accept_phrases
        bird_spotted = any(word in generated_text.lower() for word in accept_phrases)
        self.assertTrue(
            bird_spotted,
            f"Model {model_name} failed to any of {accept_phrases} in the image",
        )

        return generated_text

    ### MODEL-SPECIFIC TESTS ###
    def test_llama_3_2_vision_instruct(self):
        """Test Llama 3.2 11B Vision Instruct model"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.description_prompt}<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        self.toucan_test_runner(
            "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", prompt
        )

    def test_llama_3_2_vision_instruct_text_only(self):
        """Test Llama 3.2 11B Vision Instruct model with only text"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.text_only_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        try:
            self.toucan_test_runner(
                "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
                prompt,
                text_only=True,
            )
        except AttributeError as e:
            # mlx-lm prompt processing fails
            self.assertIn(
                "'NoneType' object has no attribute 'shape'",
                str(e),
            )

    def test_pixtral_vision(self):
        """Test Pixtral 12B model"""
        prompt = f"<s>[INST]{self.description_prompt}[IMG][/INST]"
        self.toucan_test_runner(
            "mlx-community/pixtral-12b-4bit",
            prompt,
            supplemental_accept_phrases=["bird"],
        )

    def test_pixtral_text_only(self):
        """Test Pixtral 12B model with only text"""
        prompt = f"<s>[INST]{self.text_only_prompt}[/INST]"
        self.toucan_test_runner(
            "mlx-community/pixtral-12b-4bit", prompt, text_only=True
        )

    def test_lfm2_vl_vision(self):
        """Test LFM2-VL 450M model"""
        prompt = f"""<|startoftext|><|im_start|>user
<image>{self.description_prompt}<|im_end|>
<|im_start|>assistant"""
        self.toucan_test_runner("mlx-community/LFM2-VL-450M-4bit", prompt)

    def test_lfm2_vl_text_only(self):
        """Test LFM2-VL 450M model"""
        prompt = f"""<|startoftext|><|im_start|>user
{self.text_only_prompt}<|im_end|>
<|im_start|>assistant"""
        self.toucan_test_runner(
            "mlx-community/LFM2-VL-450M-4bit", prompt, text_only=True
        )

    @pytest.mark.heavy
    def test_mistral3_vision(self):
        prompt = f"<s>[INST]{self.description_prompt}[IMG][/INST]"
        self.toucan_test_runner(
            "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit", prompt
        )

    @pytest.mark.heavy
    def test_mistral3_text_only(self):
        prompt = f"<s>[INST]{self.text_only_prompt}[IMG][/INST]"
        self.toucan_test_runner(
            "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
            prompt,
            text_only=True,
        )

    @pytest.mark.heavy
    def test_mistral3_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter(
            "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit"
        )

        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> None:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - model creates a long story
        prompt = "<s>[INST]Tell me a 500 word story[/INST]"
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(num_prompt_processing_callbacks, 2)  # single batch - 0%, 100%
        self.assertIn("clara", generated_text.lower())

        # Generation 2 - ask for a detail about the story, should not reprocess
        prompt += generated_text + "[INST]What was the main characters name?[/INST]"
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > 512 tokens cause multi-batch processing. Ensure prompt meets that condition
        self.assertGreater(num_tokens, 512)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(2, num_prompt_processing_callbacks)  # single batch - 0%, 100%
        self.assertIn("**clara**", generated_text.lower())

    def test_qwen2_vision(self):
        """Test Qwen2 VL 7B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> {self.description_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner("mlx-community/Qwen2-VL-7B-Instruct-4bit", prompt)

    def test_qwen2_text_only(self):
        """Test Qwen2 VL 7B Instruct model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen2-VL-7B-Instruct-4bit", prompt, text_only=True
        )

    def test_qwen2_5_vision(self):
        """Test Qwen2.5 VL 7B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image> {self.description_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", prompt)

    def test_qwen2_5_text_only(self):
        """Test Qwen2.5 VL 7B Instruct model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", prompt, text_only=True
        )

    def test_qwen3_vl_vision(self):
        """Test Qwen3-VL 4B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            prompt,
        )

    def test_qwen3_vl_text_only(self):
        """Test Qwen3-VL 4B Instruct model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            prompt,
            text_only=True,
        )

    @pytest.mark.heavy
    def test_qwen3_vl_moe_vision(self):
        """Test Qwen3-VL 30B-A3B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
            prompt,
        )

    @pytest.mark.heavy
    def test_qwen3_vl_moe_text_only(self):
        """Test Qwen3-VL 30B-A3B Instruct model model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
            prompt,
            text_only=True,
        )

    def test_qwen2_5_images_across_messages(self):
        model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt, images_b64):
            prompt_tokens = tokenize(model_kit, prompt)
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
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
        images_b64 = [self.toucan_image_b64]
        generated_text = generate_text(prompt, images_b64)
        # The logits for "Bird" and "T" are incredibly close to each other for this generation.
        # Therefore, accept either to reduce flakiness, as both "toucan" and "bird" are acceptable.
        acceptable_words = ["toucan", "bird"]
        is_word_accepted = any(
            word in generated_text.lower() for word in acceptable_words
        )
        self.assertTrue(
            is_word_accepted,
            f"Expected one of {acceptable_words} but got {generated_text.lower()}",
        )

        # Test case 2: Second image added in continued conversation
        prompt += """Toucan.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>In a few words, describe all of the images you've seen so far<|im_end|>
<|im_start|>assistant
"""
        images_b64 = [self.toucan_image_b64, self.chameleon_image_b64]
        generated_text = generate_text(prompt, images_b64)
        self.assertIn("toucan", generated_text.lower())
        self.assertIn("chameleon", generated_text.lower())

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_florence_vision(self):
        """Test Florence 2 Large model"""
        prompt = self.description_prompt
        self.toucan_test_runner("mlx-community/Florence-2-base-ft-4bit", prompt)

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_florence_text_only(self):
        """Test Florence 2 Large model with only text"""
        prompt = self.text_only_prompt
        try:
            self.toucan_test_runner(
                "mlx-community/Florence-2-base-ft-4bit", prompt, text_only=True
            )
        except ValueError as e:
            self.assertIn(
                "Using this model without any images attached is not supported yet.",
                str(e),
            )

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_molmo_vision(self):
        """Test Molmo 7B model"""
        prompt = self.description_prompt
        self.toucan_test_runner("mlx-community/Molmo-7B-D-0924-4bit", prompt)

    @unittest.skip("Unavailable since this requires trust_remote_code")
    def test_molmo_text_only(self):
        """Test Molmo 7B model with only text"""
        prompt = self.text_only_prompt
        self.toucan_test_runner(
            "mlx-community/Molmo-7B-D-0924-4bit", prompt, text_only=True
        )

    def test_llava_vision(self):
        """Test LLaVA v1.6 Mistral 7B model"""
        prompt = f"[INST] <image>\n{self.description_prompt} [/INST]"
        self.toucan_test_runner(
            "mlx-community/llava-v1.6-mistral-7b-4bit",
            prompt,
            supplemental_accept_phrases=["bird"],
        )

    def test_llava_text_only(self):
        """Test LLaVA v1.6 Mistral 7B model with only text"""
        prompt = f"[INST] {self.text_only_prompt} [/INST]"
        self.toucan_test_runner(
            "mlx-community/llava-v1.6-mistral-7b-4bit", prompt, text_only=True
        )

    def test_bunny_llama_vision(self):
        """Test Bunny Llama 3 8B V model"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{self.description_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        self.toucan_test_runner("mlx-community/Bunny-Llama-3-8B-V-4bit", prompt)

    def test_bunny_llama_text_only(self):
        """Test Bunny Llama 3 8B V model with only text"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.text_only_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        self.toucan_test_runner(
            "mlx-community/Bunny-Llama-3-8B-V-4bit", prompt, text_only=True
        )

    def test_nano_llava_vision(self):
        """Test Nano LLaVA 1.5 4B model"""
        prompt = f"<|im_start|>system\nAnswer the prompt.<|im_end|><|im_start|>user\n<image>\n{self.description_prompt}<|im_end|><|im_start|>assistant\n\n"
        self.toucan_test_runner("mlx-community/nanoLLaVA-1.5-4bit", prompt)

    def test_nano_llava_text_only(self):
        """Test Nano LLaVA 1.5 4B model with only text"""
        prompt = f"<|im_start|>system\nAnswer the prompt.<|im_end|><|im_start|>user\n{self.text_only_prompt}<|im_end|><|im_start|>assistant\n\n"
        self.toucan_test_runner(
            "mlx-community/nanoLLaVA-1.5-4bit", prompt, text_only=True
        )

    def test_paligemma2_vision(self):
        """Test Paligemma 2 model"""
        prompt = f"<image>{self.description_prompt}"
        self.toucan_test_runner(
            "mlx-community/paligemma2-3b-pt-896-4bit",
            prompt,
            supplemental_accept_phrases=["bird"],
        )

    def test_paligemma2_text_only(self):
        """Test Paligemma 2 model with only text"""
        try:
            prompt = self.text_only_prompt
            self.toucan_test_runner(
                "mlx-community/paligemma2-3b-pt-896-4bit", prompt, text_only=True
            )
        except ValueError as e:
            self.assertIn(
                "Using this model without any images attached is not supported yet.",
                str(e),
            )

    def test_gemma3_vision(self):
        """Test Gemma 3 model"""
        prompt = f"<bos><start_of_turn>user\n{self.description_prompt}<start_of_image><end_of_turn>\n<start_of_turn>model\n"
        self.toucan_test_runner("mlx-community/gemma-3-4b-it-4bit", prompt)

    def test_gemma3_text_only_short(self):
        """Test Gemma 3 model"""
        prompt = f"<bos><start_of_turn>user\n{self.text_only_prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.toucan_test_runner(
            "mlx-community/gemma-3-4b-it-4bit", prompt, text_only=True
        )

    def test_gemma3_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("mlx-community/gemma-3-4b-it-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> None:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - model creates a long story
        prompt = dedent("""\
            <bos><start_of_turn>user
            Tell me a 500-word story<end_of_turn>
            <start_of_turn>model
            """)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(num_prompt_processing_callbacks, 2)  # single batch - 0, 100
        self.assertIn("silas", generated_text.lower())

        # Generation 2 - ask for a detail about the story, should not reprocess
        prompt += generated_text + dedent("""\
            <end_of_turn>
            <start_of_turn>user
            What was the main characters name?<end_of_turn>
            <start_of_turn>model
            """)
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > 512 tokens cause multi-batch processing. Ensure prompt meets that condition
        self.assertGreater(num_tokens, 512)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(2, num_prompt_processing_callbacks)  # single batch - 0, 100
        self.assertIn("silas", generated_text.lower())

    def test_gemma3_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("mlx-community/gemma-3-4b-it-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> None:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - send model a long excerpt to summarize
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        # don't use dedent below b/c file content doesn't match indentation on each newline
        prompt = f"""\
<bos><start_of_turn>user
```
{file_content}
```
Summarize this in one sentence<end_of_turn>
<start_of_turn>model
"""
        num_tokens = len(model_kit.tokenize(prompt))
        self.assertGreater(num_tokens, 1024)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(
            4, num_prompt_processing_callbacks
        )  # 4 batches, so 0, x, x, 100
        self.assertIn("benjamin franklin", generated_text.lower())

        # Generation 2 - ask for a detail about the excerpt, should not reprocess
        prompt += generated_text + dedent("""\
                <end_of_turn>
                <start_of_turn>user
                What was the main characters name?<end_of_turn>
                <start_of_turn>model
                """)
        print(prompt)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(2, num_prompt_processing_callbacks)  # single batch - 0, 100
        self.assertIn("benjamin franklin", generated_text.lower())

    def test_gemma3n_vision(self):
        """Test gemma 3n model"""
        prompt = f"<bos><start_of_turn>user\n<image_soft_token>{self.description_prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.toucan_test_runner("lmstudio-community/gemma-3n-E2B-it-MLX-4bit", prompt)

    def test_gemma3n_text_only(self):
        """Test gemma 3n model text only"""
        prompt = f"<bos><start_of_turn>user\n{self.text_only_prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.toucan_test_runner(
            "lmstudio-community/gemma-3n-E2B-it-MLX-4bit", prompt, text_only=True
        )

    # TODO(will): Parameterize and de-dup
    def test_gemma3n_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> None:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - model creates a long story
        prompt = dedent("""\
            <bos><start_of_turn>user
            Tell me a 500-word story<end_of_turn>
            <start_of_turn>model
            """)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(num_prompt_processing_callbacks, 2)  # single batch - 0, 100
        self.assertIn("silas", generated_text.lower())

        # Generation 2 - ask for a detail about the story, should not reprocess
        prompt += generated_text + dedent("""\
            <end_of_turn>
            <start_of_turn>user
            What was the main characters name?<end_of_turn>
            <start_of_turn>model
            """)
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > 512 tokens cause multi-batch processing. Ensure prompt meets that condition
        self.assertGreater(num_tokens, 512)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(2, num_prompt_processing_callbacks)  # single batch - 0, 100
        self.assertIn("silas", generated_text.lower())

    # TODO(will): Parameterize and de-dup
    def test_gemma3n_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            num_prompt_processing_callbacks = 0

            def progress_callback(progress: float) -> None:
                nonlocal num_prompt_processing_callbacks
                num_prompt_processing_callbacks += 1
                print(f"Prompt processing progress: {progress}")

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_callback=progress_callback,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, num_prompt_processing_callbacks

        # Generation 1 - send model a long excerpt to summarize
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        # don't use dedent below b/c file content doesn't match indentation on each newline
        prompt = f"""\
<bos><start_of_turn>user
```
{file_content}
```
Summarize this in one sentence<end_of_turn>
<start_of_turn>model
"""
        num_tokens = len(model_kit.tokenize(prompt))
        self.assertGreater(num_tokens, 1024)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(
            4, num_prompt_processing_callbacks
        )  # 4 batches, so 0, x, x, 100
        self.assertIn("benjamin franklin", generated_text.lower())

        # Generation 2 - ask for a detail about the excerpt, should not reprocess
        prompt += generated_text + dedent("""\
                <end_of_turn>
                <start_of_turn>user
                What was the main characters name?<end_of_turn>
                <start_of_turn>model
                """)
        print(prompt)
        generated_text, num_prompt_processing_callbacks = generate_text(prompt)
        self.assertEqual(2, num_prompt_processing_callbacks)  # single batch - 0, 100
        self.assertIn("benjamin franklin", generated_text.lower())

    def test_gemma3n_vision_long_prompt_progress_reported(self):
        """Ensure progress is reported during prompt processing with a vision prompt"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        # don't use dedent below b/c file content doesn't match indentation on each newline
        prompt = f"""\
<bos><start_of_turn>user
<image_soft_token>
```
{file_content}
```
Summarize this in one sentence<end_of_turn>
<start_of_turn>model
"""
        prompt_tokens = tokenize(model_kit, prompt)
        progress_values = []

        def progress_callback(progress: float) -> bool:
            nonlocal progress_values
            progress_values.append(progress)
            print(f"Prompt processing progress: {progress}")
            return True

        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=[self.toucan_image_b64],
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
        self.assertGreater(len(progress_values), 0)
        for i in range(len(progress_values) - 1):
            self.assertGreater(progress_values[i + 1], progress_values[i])

    ### NON-MODEL-SPECIFIC TESTS ###
    def test_draft_model_not_compatible_vision(self):
        model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = load_model(model_path=model_path)
        self.assertFalse(
            is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)
        )


"""
To find the correct prompt format for new models, run this command for your model in the terminal and check the prompt dump:
python -m mlx_vlm.generate --model ~/.cache/lm-studio/models/mlx-community/MODEL-NAME --max-tokens 100 --temp 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg --prompt "What do you see?"
"""
