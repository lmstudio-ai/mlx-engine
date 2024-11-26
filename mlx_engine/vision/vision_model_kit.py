from typing import Union, Optional, List

from mlx_engine.model_kit import ModelKit
from .vision_model_wrapper import VisionModelWrapper

import mlx_vlm
from pathlib import Path
import mlx.core as mx
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class VisionModelKit(ModelKit):
    """
    Collection of objects and methods that are needed for operating a vision model
    """

    config: dict = None
    trust_remote_code: bool = False
    model_path: Path = None
    max_kv_size: int = None

    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    has_processed_prompt: bool = False

    def __init__(self, model_path: Path, max_kv_size: int, trust_remote_code: bool):
        self.config = mlx_vlm.utils.load_config(model_path)
        self.trust_remote_code = trust_remote_code
        self.model_path = model_path
        self.max_kv_size = max_kv_size
        self._initializer()

    def _initializer(self):
        self.model, self.processor = mlx_vlm.utils.load(
            self.model_path,
            processor_config={"trust_remote_code": self.trust_remote_code},
        )
        image_processor = mlx_vlm.utils.load_image_processor(self.model_path)
        self.model = VisionModelWrapper(self.model, image_processor)
        self.tokenizer = mlx_vlm.tokenizer_utils.load_tokenizer(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = None
        mx.metal.clear_cache()

    def _reset(self):
        # it's a shortcoming that the only way to reset the model is to reload it
        # worth investigating how to make resetting faster
        self._initializer()

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_callback,
        repetition_context_size,
        generate_args,
    ) -> mx.array:
        """
        Call this before starting evaluation

        This method opens the image from the base64-encoded string, and adds the special image token to the prompt

        Returns the input for the `generate_step` function
        """
        if self.has_processed_prompt:
            self._reset()

        self.model.process_prompt_with_images(
            images_b64, prompt_tokens, self.processor, self.detokenizer
        )
        self.has_processed_prompt = True

        # disable `prefill_step_size` prompt pre-processing in mlx_lm::generate_step
        generate_args["prefill_step_size"] = float("inf")

        generate_step_input = self.model.input_ids[None]
        return generate_step_input

    def record_generated_token(self, token: int) -> None:
        pass

    def record_sampled_token(self, token: int) -> None:
        self.model.record_sampled_token(token)

    @property
    def language_model(self):
        return self.model.language_model
