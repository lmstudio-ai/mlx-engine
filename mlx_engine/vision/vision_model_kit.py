from typing import Union

from mlx_engine.model_kit import ModelKit
from .vision_model_wrapper import VisionModelWrapper

import PIL
from io import BytesIO
import base64
import mlx_vlm
from pathlib import Path
import mlx.core as mx
from mlx_vlm.prompt_utils import get_message_json
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class VisionModelKit(ModelKit):
    """
    Collection of objects and methods that are needed for operating a vision model
    """

    config: dict = None
    trust_remote_code: bool = False
    model_path: Path = None

    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    has_processed_prompt: bool = False

    def __init__(self, model_path: str, trust_remote_code: bool):
        self.config = mlx_vlm.utils.load_config(model_path)
        self.trust_remote_code = trust_remote_code
        self.model_path = Path(model_path)
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

    def _reset(self):
        # it's a shortcoming that the only way to reset the model is to reload it
        # worth investigating how to make resetting faster
        self._initializer()

    def process_prompt(
        self, prompt_tokens, img_b64, prompt_progress_callback, generate_args
    ) -> mx.array:
        """
        Call this before starting evaluation

        This method opens the image from the base64-encoded string, and adds the special image token to the prompt

        Returns the input for the `generate_step` function
        """
        if self.has_processed_prompt:
            self._reset()

        image = PIL.Image.open(BytesIO(base64.b64decode(img_b64))) if img_b64 else None
        try:
            image_token_format = get_message_json(self.config["model_type"], "")[
                "content"
            ]
        except:
            raise ValueError("Model type is not supported")
        self.model.process_prompt_with_image(
            image, prompt_tokens, self.processor, self.detokenizer, image_token_format
        )
        self.has_processed_prompt = True

        # disable `prefill_step_size` prompt pre-processing in mlx_lm::generate_step
        generate_args["prefill_step_size"] = float("inf")

        generate_step_input = self.model.input_ids[0]
        return generate_step_input

    @property
    def language_model(self):
        return self.model.language_model
