import json
from typing import Union, Optional, List

from mlx_engine.model_kit import ModelKit
from mlx_engine.logging import log_info, log_warn
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
    vocab_only: bool = False
    model_weights = None

    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    has_processed_prompt: bool = False

    def __init__(
        self,
        model_path: Path,
        vocab_only: bool,
        trust_remote_code: bool,
    ):
        self._fix_qwen2_5_vl_image_processor(model_path)
        self.config = mlx_vlm.utils.load_config(
            model_path, trust_remote_code=trust_remote_code
        )
        self.trust_remote_code = trust_remote_code
        self.vocab_only = vocab_only
        self.model_path = model_path
        self._initializer()

    @staticmethod
    def _fix_qwen2_5_vl_image_processor(model_path: Path):
        """
        Register Qwen2_5_VLImageProcessor as Qwen2VLImageProcessor in AutoImageProcessor
        This is needed because Qwen2_5_VLImageProcessor was deleted from transformers
        Ref https://github.com/Blaizzy/mlx-vlm/issues/209#issuecomment-2678113857
        Ref https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit/commit/fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b
        """
        try:
            # We are looking for a specific entry, so if any of this throws, we don't need to do anything
            with open(model_path / "preprocessor_config.json", "r") as f:
                image_processor_type = json.load(f)["image_processor_type"]
        except:
            return

        if image_processor_type != "Qwen2_5_VLImageProcessor":
            return

        log_info(
            f"Registering deprecated Qwen2_5_VLImageProcessor as Qwen2VLImageProcessor"
        )
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                Qwen2VLImageProcessor,
            )
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
                Qwen2_5_VLConfig,
            )
            from transformers.models.auto.processing_auto import AutoImageProcessor

            class Qwen2_5_VLImageProcessor(Qwen2VLImageProcessor):
                pass

            AutoImageProcessor.register(
                Qwen2_5_VLConfig, Qwen2_5_VLImageProcessor, exist_ok=True
            )
        except Exception as e:
            log_warn(
                f"Failed to register Qwen2_5_VLImageProcessor to AutoImageProcessor: {e}"
            )

    def _vocab_only_init(self):
        self.tokenizer = mlx_vlm.tokenizer_utils.load_tokenizer(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer

    def _full_model_init(self):
        additional_kwargs = {}
        if self.model_weights:
            additional_kwargs["weights"] = self.model_weights
        return_tuple = mlx_vlm.utils.load(
            self.model_path,
            processor_config={"trust_remote_code": self.trust_remote_code},
            trust_remote_code=self.trust_remote_code,
            **additional_kwargs,
        )
        if len(return_tuple) == 2:
            self.model, self.processor = return_tuple
        else:
            self.model, self.processor, self.model_weights = return_tuple
        self.model = VisionModelWrapper(self.model)
        self.tokenizer = mlx_vlm.tokenizer_utils.load_tokenizer(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = None
        mx.metal.clear_cache()

    def _initializer(self):
        if self.vocab_only:
            self._vocab_only_init()
        else:
            self._full_model_init()

    def _reset_for_prediction(self):
        # It's a shortcoming that the only way to reset the model for prediction
        # is to reload it. Worth investigating how to make resetting faster
        self._full_model_init()

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_callback,
        repetition_context_size,
        generate_args,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> mx.array:
        """
        Call this before starting evaluation

        This method opens the image from the base64-encoded string, and adds the special image token to the prompt

        Returns the input for the `generate_step` function
        """
        if self.has_processed_prompt:
            self._reset_for_prediction()

        self.model.process_prompt_with_images(
            images_b64, prompt_tokens, self.processor, self.detokenizer
        )
        self.has_processed_prompt = True

        # disable `prefill_step_size` prompt pre-processing in mlx_lm::generate_step
        generate_args["prefill_step_size"] = float("inf")

        # The VLM input_ids shape is important, but mlx_lm expects a flattened array
        #   Send the prompt back reshaped, and save the real shape in `self.model.input_ids`
        return self.model.input_ids.reshape(-1)

    def update_cache_wrapper(self, token: int) -> None:
        pass

    def record_sampled_token(self, token: int) -> None:
        self.model.record_sampled_token(token)

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        return False

    def load_draft_model(self, path: str | Path) -> None:
        raise ValueError(
            "Speculative decoding is not currently supported for vision models"
        )

    def unload_draft_model(self) -> None:
        raise ValueError(
            "Speculative decoding is not currently supported for vision models"
        )

    @property
    def language_model(self):
        return self.model.language_model
