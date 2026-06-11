from pathlib import Path
import pytest
import mlx.core as mx
from mlx_engine.generate import (
    load_model,
    tokenize,
    create_generator,
    is_draft_model_compatible,
    unload,
)
from mlx_engine.model_kit.batched_vision import BatchedVisionModelKit
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    build_prompt_kwargs,
    prepare_prompt_inputs,
)
from tests.shared import (
    model_getter,
    RecordingReporter,
    read_image_b64,
    model_load_and_tokenize_prompt,
)
from textwrap import dedent
from transformers import AutoProcessor


MAX_KV_CACHE_SIZE = 20000

# 512 was the previous default, and some tests were written with assertions
# on the number of prompt processing events.
CACHING_TEST_PREFILL_STEP_SIZE = 512


def _assert_batched_vlm_reporter_events(reporter: RecordingReporter) -> None:
    """Assert basic progress invariants for the batched VLM path."""
    assert reporter.events
    begin_event = reporter.events[0]
    finish_event = reporter.events[-1]
    assert begin_event["type"] == "begin"
    assert finish_event["type"] == "finish"
    assert begin_event["prefill_tokens_processed"] == 0
    assert 0 <= begin_event["cached_tokens"] <= begin_event["total_prompt_tokens"]

    progress_values = [
        begin_event["prefill_tokens_processed"],
        *[
            event["prefill_tokens_processed"]
            for event in reporter.events
            if event["type"] == "update"
        ],
        finish_event["prefill_tokens_processed"],
    ]
    assert progress_values == sorted(progress_values)
    assert finish_event["prefill_tokens_processed"] >= 0


def _expected_cached_text_prompt_tokens(model_kit, prompt: str) -> set[int]:
    token_count = len(model_kit.tokenize(prompt))
    # The runtime cache may include the sampled stop token; generated_text does not.
    return {token_count, token_count + 1}


def _assert_cached_text_prompt_reused(
    begin_event: dict, expected_cached_tokens: set[int]
) -> None:
    assert begin_event["cached_tokens"] >= CACHING_TEST_PREFILL_STEP_SIZE
    assert begin_event["cached_tokens"] <= max(expected_cached_tokens)


def _assert_cached_follow_up_prefill_is_small(
    begin_event: dict,
    finish_event: dict,
) -> None:
    assert finish_event["type"] == "finish"
    assert finish_event["prefill_tokens_processed"] == (
        begin_event["total_prompt_tokens"] - begin_event["cached_tokens"]
    )
    assert finish_event["prefill_tokens_processed"] <= CACHING_TEST_PREFILL_STEP_SIZE


def _assert_mentions_franklin(text: str) -> None:
    assert "franklin" in text.lower()


def _assert_ready_only(text: str) -> None:
    assert " ".join(text.strip().lower().split()) == "ready"


class TestVisionModels:
    @classmethod
    def setup_class(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.description_prompt = "What is this"
        cls.text_only_prompt = "What is a toucan?"
        cls.test_data_dir = Path(__file__).parent / "data"
        cls.demo_data_dir = Path(__file__).parent.parent / "demo-data"

        # Read and encode test images
        cls.toucan_path = Path(__file__).parent.parent / "demo-data" / "toucan.jpeg"
        cls.toucan_image_b64 = read_image_b64(cls.toucan_path)
        cls.chameleon_image_path = (
            Path(__file__).parent.parent / "demo-data" / "chameleon.webp"
        )
        cls.chameleon_image_b64 = read_image_b64(cls.chameleon_image_path)

    def setup_method(self):
        self._loaded_model_kit = None

    def teardown_method(self):
        if self._loaded_model_kit is not None:
            unload(self._loaded_model_kit)

    def load_model(self, *args, **kwargs):
        assert self._loaded_model_kit is None
        model_kit = load_model(*args, **kwargs)
        self._loaded_model_kit = model_kit
        return model_kit

    def toucan_test_runner(
        self,
        model_name: str,
        prompt: str,
        text_only=False,
        supplemental_accept_phrases=None,
        max_tokens=30,
    ):
        """Helper method to test a single vision model"""
        print(f"Testing model {model_name}")
        model_path = model_getter(model_name=model_name)

        # Load the model
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=2048,
            trust_remote_code=True,
        )
        assert isinstance(model_kit, BatchedVisionModelKit)

        try:
            # Tokenize the prompt
            prompt_tokens = tokenize(model_kit, prompt)

            # Generate description
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=([self.toucan_image_b64] if not text_only else None),
                seed=0,
                max_tokens=max_tokens,
                temp=0.0,
                repetition_penalty=1.01,  # enable the logits processor code path
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
            bird_spotted = any(
                word in generated_text.lower() for word in accept_phrases
            )
            assert bird_spotted, (
                f"Model {model_name} failed to any of {accept_phrases} in the image"
            )

            return generated_text
        finally:
            unload(model_kit)

    def build_gemma4_prompt(
        self,
        model_path: Path,
        prompt: str,
        *,
        text_only: bool = False,
    ) -> str:
        processor = AutoProcessor.from_pretrained(model_path)
        content = [{"type": "text", "text": prompt}]
        if not text_only:
            content.insert(0, {"type": "image", "base64": self.toucan_image_b64})
        conversation = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_granite4_prompt(self, model_path: Path, prompt: str) -> str:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    ### MODEL-SPECIFIC TESTS ###
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

    def test_lfm2_5_vl_vision(self):
        """Test LFM2.5-VL 1.6B model"""
        prompt = f"""<|im_start|>user
<image>{self.description_prompt}<|im_end|>
<|im_start|>assistant
"""
        self.toucan_test_runner("lmstudio-community/LFM2.5-VL-1.6B-MLX-4bit", prompt)

    # This test was added due to a failure observed with mlx-vlm be852ea
    # ref: https://github.com/Blaizzy/mlx-vlm/issues/698#issuecomment-3887073430
    def test_lfm2_5_vl_vision_equations(self):
        prompt = """<|im_start|>user
<image>What is this<|im_end|>
<|im_start|>assistant
"""

        model_kit, prompt_tokens = model_load_and_tokenize_prompt(
            model_name="lmstudio-community/LFM2.5-VL-1.6B-MLX-4bit",
            prompt=prompt,
            trust_remote_code=False,
        )

        equations_image_path = self.test_data_dir / "equations.jpg"
        equations_image_b64 = read_image_b64(equations_image_path)

        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=[equations_image_b64],
            seed=0,
            max_tokens=30,
            temp=0.0,
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print()
        # Simply ensure we got here without error, for now.

    def test_lfm2_5_vl_text_only(self):
        """Test LFM2.5-VL 1.6B model"""
        prompt = f"""<|im_start|>user
{self.text_only_prompt}<|im_end|>
<|im_start|>assistant
"""
        self.toucan_test_runner(
            "lmstudio-community/LFM2.5-VL-1.6B-MLX-4bit", prompt, text_only=True
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
    @pytest.mark.parametrize(
        "model",
        [
            "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
            "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
        ],
    )
    def test_mistral3_text_only_generation_caching(self, model):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter(model)

        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        # Generation 1 - model creates a long story
        prompt = "<s>[INST]Tell me a 500 word story about the bravest soul in the middle ages, and their weapon of choice[/INST]"
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        assert "aldric" in generated_text.lower()

        # Generation 2 - ask for a detail about the story, should not reprocess
        cached_prompt = prompt + generated_text
        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, cached_prompt
        )
        prompt = cached_prompt + "[INST]What was the main character's name?[/INST]"
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > prefill_step_size tokens cause multi-chunk processing.
        assert num_tokens > CACHING_TEST_PREFILL_STEP_SIZE
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] in expected_cached_tokens
        assert "aldric" in generated_text.lower()

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
            "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
            prompt,
        )

    def test_qwen3_vl_text_only(self):
        """Test Qwen3-VL 4B Instruct model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
            prompt,
            text_only=True,
        )

    @pytest.mark.heavy
    def test_qwen3_vl_moe_vision(self):
        """Test Qwen3-VL 30B-A3B Instruct model"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit",
            prompt,
        )

    @pytest.mark.heavy
    def test_qwen3_vl_moe_text_only(self):
        """Test Qwen3-VL 30B-A3B Instruct model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit",
            prompt,
            text_only=True,
        )

    def test_qwen2_5_images_across_messages(self):
        model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
        )

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
        assert is_word_accepted, (
            f"Expected one of {acceptable_words} but got {generated_text.lower()}"
        )

        # Test case 2: Second image added in continued conversation
        prompt += """Toucan.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>In a few words, describe all of the images you've seen so far<|im_end|>
<|im_start|>assistant
"""
        images_b64 = [self.toucan_image_b64, self.chameleon_image_b64]
        generated_text = generate_text(prompt, images_b64)
        assert "toucan" in generated_text.lower()
        assert "chameleon" in generated_text.lower()

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
            assert (
                "Using this model without any images attached is not supported yet."
                in str(e)
            )

    def test_gemma3_vision(self):
        """Test Gemma 3 model"""
        description_prompt = (
            "Describe this image in 2-3 short sentences. Mention the bird, "
            "its large colorful beak, and the natural photographic tone. "
            "Do not use bullet points."
        )
        prompt = f"<bos><start_of_turn>user\n{description_prompt}<start_of_image><end_of_turn>\n<start_of_turn>model\n"
        generated_text = self.toucan_test_runner(
            "mlx-community/gemma-3-4b-it-4bit",
            prompt,
            max_tokens=80,
        )
        assert any(word in generated_text.lower() for word in ["beak", "bill"])
        assert not any(
            line.lstrip().startswith(("-", "*", "•"))
            for line in generated_text.splitlines()
        )

    def test_gemma3_text_only_short(self):
        """Test Gemma 3 model"""
        prompt = f"<bos><start_of_turn>user\n{self.text_only_prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.toucan_test_runner(
            "mlx-community/gemma-3-4b-it-4bit", prompt, text_only=True
        )

    def test_gemma3_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("mlx-community/gemma-3-4b-it-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        # Generation 1 - model creates a long story
        prompt = dedent("""\
            <bos><start_of_turn>user
            Tell me a 500-word story about a main character named Silas<end_of_turn>
            <start_of_turn>model
            """)
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        assert "silas" in generated_text.lower()

        # Generation 2 - ask for a detail about the story, should not reprocess
        cached_prompt = prompt + generated_text
        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, cached_prompt
        )
        prompt = cached_prompt + dedent("""\
            <end_of_turn>
            <start_of_turn>user
            What was the main characters name?<end_of_turn>
            <start_of_turn>model
            """)
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > prefill_step_size tokens cause multi-chunk processing.
        assert num_tokens > CACHING_TEST_PREFILL_STEP_SIZE
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        _assert_cached_text_prompt_reused(begin_event, expected_cached_tokens)
        assert "silas" in generated_text.lower()

    def test_gemma3_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("mlx-community/gemma-3-4b-it-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )
        print(type(model_kit))

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        # Generation 1 - send model a long excerpt and cache it.
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        # don't use dedent below b/c file content doesn't match indentation on each newline
        prompt = f"""\
<bos><start_of_turn>user
Read the excerpt below. When you are done, reply with exactly READY and nothing else.

```
{file_content}
```
Reply with exactly READY and nothing else.<end_of_turn>
<start_of_turn>model
"""
        num_tokens = len(model_kit.tokenize(prompt))
        assert num_tokens > 1024
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        _assert_ready_only(generated_text)

        # Generation 2 - ask for a detail about the excerpt, should not reprocess
        cached_prompt = prompt + generated_text
        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, cached_prompt
        )
        prompt = cached_prompt + dedent("""\
                <end_of_turn>
                <start_of_turn>user
                Who is the author of this passage? Answer with only the name.<end_of_turn>
                <start_of_turn>model
                """)
        print(prompt)
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        _assert_cached_text_prompt_reused(begin_event, expected_cached_tokens)
        _assert_mentions_franklin(generated_text)

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

    @pytest.mark.heavy
    def test_gemma4_vision(self):
        """Test Gemma 4 model via the unified multimodal path."""
        model_name = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
        model_path = model_getter(model_name)
        prompt = self.build_gemma4_prompt(model_path, self.description_prompt)
        self.toucan_test_runner(
            model_name,
            prompt,
            supplemental_accept_phrases=["bird"],
        )

    def test_granite4_vision(self):
        """Test Granite 4 Vision model."""
        model_name = "mlx-community/granite-4.0-3b-vision-4bit"
        model_path = model_getter(model_name)
        prompt = self.build_granite4_prompt(model_path, self.description_prompt)
        self.toucan_test_runner(
            model_name,
            prompt,
            supplemental_accept_phrases=["bird"],
        )

    @pytest.mark.heavy
    def test_gemma4_text_only(self):
        """Test Gemma 4 model with text only via the unified multimodal path."""
        model_name = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
        model_path = model_getter(model_name)
        prompt = self.build_gemma4_prompt(
            model_path,
            self.text_only_prompt,
            text_only=True,
        )
        self.toucan_test_runner(model_name, prompt, text_only=True)

    @pytest.mark.heavy
    def test_gemma4_text_only_generation_caching(self):
        """Gemma 4 VLM progress reports cached and uncached prompt tokens."""
        model_name = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
        model_path = model_getter(model_name)
        processor = AutoProcessor.from_pretrained(model_path)
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )

        def render_prompt(conversation):
            return processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        first_conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Tell me a 500-word story about a traveler named Silas."
                        ),
                    }
                ],
            }
        ]

        prompt = render_prompt(first_conversation)
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        assert len(generated_text) > 0
        assert "silas" in generated_text.lower()

        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, prompt + generated_text
        )
        second_conversation = first_conversation + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What was the main character's name? Answer with only the name.",
                    }
                ],
            },
        ]
        prompt = render_prompt(second_conversation)
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, the follow-up prompt is long enough to require multi-chunk prefill.
        assert num_tokens > CACHING_TEST_PREFILL_STEP_SIZE

        follow_up_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        _assert_cached_text_prompt_reused(begin_event, expected_cached_tokens)
        finish_event = reporter.events[-1]
        _assert_cached_follow_up_prefill_is_small(begin_event, finish_event)
        assert "silas" in follow_up_text.lower()

    @pytest.mark.heavy
    def test_gemma4_scratchpad_follow_up_reuses_checkpoint_cache(self):
        """Gemma 4 should reuse an image checkpoint after another request."""
        model_name = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
        model_path = model_getter(model_name)
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )
        sliding_window_size = int(model_kit.config["text_config"]["sliding_window"])
        control_word = "MERIDIAN"
        background_words = (
            (self.test_data_dir / "ben_franklin_autobiography_start.txt")
            .read_text()
            .split()
        )

        def generate_text(prompt, *, max_tokens, images_b64=None):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                seed=0,
                temp=0.0,
                max_tokens=max_tokens,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return prompt_tokens, generated_text, reporter

        background = " ".join(background_words[:384])
        first_user_text = dedent(f"""\
            Read the image and background carefully.

            Background:
            {background}

            Write two sections in order.
            First, write a long SCRATCHPAD section with at least 60 numbered lines.
            Keep each line short, but make sure the full scratchpad is detailed.
            Each line should briefly analyze or restate part of the background.
            Second, end with a single final line in the exact format:
            FINAL: {control_word}
            Do not write anything after that final line.
            """).strip()

        # Own this prompt shape explicitly. mlx-vlm's Gemma4 processor renderer
        # reinserts all images into the latest user turn, but this cache test is
        # about a stable rendered prefix with the image in the original turn.
        first_prompt = (
            "<bos><|turn>user\n"
            + "<|image|>"
            + first_user_text
            + "<turn|>\n"
            + "<|turn>model\n"
        )

        (
            _,
            generated_text,
            reporter,
        ) = generate_text(
            first_prompt,
            max_tokens=1600,
            images_b64=[self.toucan_image_b64],
        )
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        first_total_prompt_tokens = begin_event["total_prompt_tokens"]
        assert first_total_prompt_tokens > CACHING_TEST_PREFILL_STEP_SIZE
        assert len(generated_text) > 0

        final_marker = f"FINAL: {control_word}"
        final_index = generated_text.find(final_marker)
        assert final_index >= 0, generated_text
        scratchpad_text = generated_text[:final_index]
        scratchpad_token_count = len(
            model_kit.tokenizer.encode(scratchpad_text, add_special_tokens=False)
        )
        assert scratchpad_token_count > sliding_window_size

        # Replace the one-entry hot cache so the follow-up must use disk cache.
        unrelated_prompt = "<bos><|turn>user\nReply OK.<turn|>\n<|turn>model\n"
        _, unrelated_text, _ = generate_text(unrelated_prompt, max_tokens=8)
        assert len(unrelated_text.strip()) > 0

        assistant_text = control_word
        follow_up_question = (
            "What was the exact single-word final answer from your previous "
            "message? Reply with only that word."
        )

        second_prompt = (
            first_prompt
            + assistant_text
            + "<turn|>\n"
            + "<|turn>user\n"
            + follow_up_question
            + "<turn|>\n"
            + "<|turn>model\n"
        )

        _, follow_up_text, reporter = generate_text(
            second_prompt,
            max_tokens=64,
            images_b64=[self.toucan_image_b64],
        )
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] >= CACHING_TEST_PREFILL_STEP_SIZE
        assert begin_event["cached_tokens"] <= first_total_prompt_tokens
        assert begin_event["cached_tokens"] < begin_event["total_prompt_tokens"]
        finish_event = reporter.events[-1]
        _assert_cached_follow_up_prefill_is_small(begin_event, finish_event)
        assert control_word.lower() in follow_up_text.lower()

    # TODO(will): Parameterize and de-dup
    def test_gemma3n_text_only_generation_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        # Generation 1 - model creates a long story
        prompt = dedent("""\
            <bos><start_of_turn>user
            Tell me a 500-word story about a main character named Silas<end_of_turn>
            <start_of_turn>model
            """)
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        assert "silas" in generated_text.lower()

        # Generation 2 - ask for a detail about the story, should not reprocess
        cached_prompt = prompt + generated_text
        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, cached_prompt
        )
        prompt = cached_prompt + dedent("""\
            <end_of_turn>
            <start_of_turn>user
            What was the main characters name?<end_of_turn>
            <start_of_turn>model
            """)
        num_tokens = len(model_kit.tokenize(prompt))
        # Without caching, prompts > prefill_step_size tokens cause multi-chunk processing.
        assert num_tokens > CACHING_TEST_PREFILL_STEP_SIZE
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        _assert_cached_text_prompt_reused(begin_event, expected_cached_tokens)
        assert "silas" in generated_text.lower()

    # TODO(will): Parameterize and de-dup
    def test_gemma3n_text_only_long_original_prompt_caching(self):
        """Ensure that text only prompts with vlms take full advantage of caching generated tokens"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
            prefill_step_size=CACHING_TEST_PREFILL_STEP_SIZE,
        )

        def generate_text(prompt):
            prompt_tokens = tokenize(model_kit, prompt)
            reporter = RecordingReporter()

            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                temp=0.0,
                max_tokens=1000,
                repetition_penalty=1.01,  # to enable this code path
                prompt_progress_reporter=reporter,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print("\n", flush=True)
            return generated_text, reporter

        # Generation 1 - send model a long excerpt and cache it.
        file_path = self.test_data_dir / "ben_franklin_autobiography_start.txt"
        file_content = file_path.read_text()
        # don't use dedent below b/c file content doesn't match indentation on each newline
        prompt = f"""\
<bos><start_of_turn>user
Read the excerpt below. When you are done, reply with exactly READY and nothing else.

```
{file_content}
```
Reply with exactly READY and nothing else.<end_of_turn>
<start_of_turn>model
"""
        num_tokens = len(model_kit.tokenize(prompt))
        assert num_tokens > 1024
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        assert begin_event["cached_tokens"] == 0
        _assert_ready_only(generated_text)

        # Generation 2 - ask for a detail about the excerpt, should not reprocess
        cached_prompt = prompt + generated_text
        expected_cached_tokens = _expected_cached_text_prompt_tokens(
            model_kit, cached_prompt
        )
        prompt = cached_prompt + dedent("""\
                <end_of_turn>
                <start_of_turn>user
                Who is the author of this passage? Answer with only the name.<end_of_turn>
                <start_of_turn>model
                """)
        print(prompt)
        generated_text, reporter = generate_text(prompt)
        _assert_batched_vlm_reporter_events(reporter)
        begin_event = reporter.events[0]
        assert begin_event["type"] == "begin"
        _assert_cached_text_prompt_reused(begin_event, expected_cached_tokens)
        finish_event = reporter.events[-1]
        _assert_cached_follow_up_prefill_is_small(begin_event, finish_event)
        _assert_mentions_franklin(generated_text)

    def test_gemma3n_vision_long_prompt_progress_reported(self):
        """Ensure progress is reported during prompt processing with a vision prompt"""
        model_path = model_getter("lmstudio-community/gemma-3n-E2B-it-MLX-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=MAX_KV_CACHE_SIZE,
        )

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
        reporter = RecordingReporter()

        generated_text = ""
        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            images_b64=[self.toucan_image_b64],
            seed=0,
            temp=0.0,
            max_tokens=1,  # We only care about pre-fill in this test
            repetition_penalty=1.01,
            prompt_progress_reporter=reporter,
        ):
            generated_text += result.text
            print(result.text, end="", flush=True)
            if result.stop_condition:
                break
        print("\n", flush=True)

        # Extract progress values from update events
        progress_values = [
            event["prefill_tokens_processed"]
            for event in reporter.events
            if event["type"] == "update"
        ]
        print(progress_values)
        assert len(progress_values) > 0
        for i in range(len(progress_values) - 1):
            assert progress_values[i + 1] > progress_values[i]

    def test_qwen3_5_vision(self):
        """Test Qwen3.5 2B model with vision"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3.5-2B-MLX-4bit",
            prompt,
        )

    def test_qwen3_5_text_only(self):
        """Test Qwen3.5 2B model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3.5-2B-MLX-4bit",
            prompt,
            text_only=True,
        )

    def test_qwen3_5_vision_then_text_only(self):
        """Test that text-only generation after a vision request produces the
        same output as a cold-start text-only generation, verifying that MRoPE
        state from the vision request does not leak into subsequent text-only
        requests."""
        model_path = model_getter("lmstudio-community/Qwen3.5-2B-MLX-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=2048,
            trust_remote_code=True,
        )

        def generate_text(prompt, images_b64=None):
            prompt_tokens = tokenize(model_kit, prompt)
            generated_text = ""
            for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                seed=0,
                temp=0.0,
                max_tokens=30,
                repetition_penalty=1.01,
            ):
                generated_text += result.text
                print(result.text, end="", flush=True)
                if result.stop_condition:
                    break
            print()
            return generated_text

        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        vision_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Step 1: Text-only baseline (cold start)
        baseline_text = generate_text(text_prompt)

        # Step 2: Confirm determinism with a second text-only run
        second_text = generate_text(text_prompt)
        assert baseline_text == second_text, (
            f"Text-only generation is not deterministic: {repr(baseline_text)} != {repr(second_text)}"
        )

        # Step 3: Vision request (populates MRoPE state)
        vision_text = generate_text(vision_prompt, images_b64=[self.toucan_image_b64])
        assert len(vision_text) > 0

        # Step 4: Text-only after vision — must match baseline exactly
        after_vision_text = generate_text(text_prompt)
        assert baseline_text == after_vision_text, (
            f"Text-only output after vision request differs from baseline "
            f"(MRoPE state likely leaked): {repr(baseline_text)} != {repr(after_vision_text)}"
        )

    def test_qwen3_5_multi_image_process_prompt_preserves_image_positions(self):
        """Qwen3.5 must inject MRoPE positions for every image span.

        This targets the prompt-processing path directly instead of relying on
        generation quality: after two images across messages, both expanded
        image token runs should still carry image-style position IDs rather than
        falling back to sequential text positions.
        """
        model_path = model_getter("lmstudio-community/Qwen3.5-2B-MLX-4bit")
        model_kit = self.load_model(
            model_path=model_path,
            max_kv_size=2048,
            trust_remote_code=True,
        )

        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is this? In one word.<|im_end|>
<|im_start|>assistant
Toucan.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>In a few words, describe all of the images you've seen so far<|im_end|>
<|im_start|>assistant
"""

        prompt_tokens = tokenize(model_kit, prompt)
        assert isinstance(model_kit, BatchedVisionModelKit)
        prepared_prompt = prepare_prompt_inputs(
            prompt_tokens=prompt_tokens,
            images_b64=[self.toucan_image_b64, self.chameleon_image_b64],
            tokenizer=model_kit.tokenizer,
            processor=model_kit.processor,
            config=model_kit.config,
        )
        prompt_kwargs = build_prompt_kwargs(model_kit.model, prepared_prompt)
        mx.eval(prompt_kwargs["inputs_embeds"])

        position_ids = prompt_kwargs["position_ids"]
        image_spans = prepared_prompt.image_spans

        assert len(image_spans) == 2, (
            f"Expected two expanded image spans in the prompt, got {image_spans!r}"
        )

        for span_idx, span in enumerate(image_spans):
            temporal_positions = position_ids[0, 0, span.start : span.end].tolist()
            assert len(set(temporal_positions)) == 1, (
                f"Image span {span_idx} used sequential text positions instead of "
                f"image MRoPE positions: {temporal_positions[:12]!r}..."
            )

    @pytest.mark.heavy
    def test_qwen3_5_moe_vision(self):
        """Test Qwen3.5 35B-A3B MoE model with vision"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{self.description_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3.5-35B-A3B-MLX-4bit",
            prompt,
        )

    @pytest.mark.heavy
    def test_qwen3_5_moe_text_only(self):
        """Test Qwen3.5 35B-A3B MoE model with only text"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{self.text_only_prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.toucan_test_runner(
            "lmstudio-community/Qwen3.5-35B-A3B-MLX-4bit",
            prompt,
            text_only=True,
        )

    ### NON-MODEL-SPECIFIC TESTS ###
    def test_draft_model_not_compatible_vision(self):
        model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        draft_model_path = model_getter(
            "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
        )
        model_kit = self.load_model(model_path=model_path)
        assert not is_draft_model_compatible(model_kit=model_kit, path=draft_model_path)

    @pytest.mark.heavy
    def test_devstral_small_2_vision(self):
        """Test Devstral Small 2 model"""
        prompt = f"<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST][IMG]{self.description_prompt}[/INST]"
        self.toucan_test_runner(
            "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", prompt
        )

    @pytest.mark.heavy
    def test_devstral_small_2_text_only(self):
        """Test Devstral Small 2 model with text only"""
        prompt = f"<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST][IMG]{self.text_only_prompt}[/INST]"
        self.toucan_test_runner(
            "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
            prompt,
            text_only=True,
        )


"""
To find the correct prompt format for new models, run this command for your model in the terminal and check the prompt dump:
python -m mlx_vlm.generate --model ~/.cache/lm-studio/models/mlx-community/MODEL-NAME --max-tokens 100 --temp 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg --prompt "What do you see?"
"""
