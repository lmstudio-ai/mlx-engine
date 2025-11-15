from typing import Union, Optional, Tuple
from mlx_engine.model_kit.model_kit import ModelKit
import logging

from ._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)
from .vision_model_wrapper import VisionModelWrapper
import mlx_vlm
import mlx_lm
from pathlib import Path
import mlx.core as mx
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


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
        fix_qwen2_5_vl_image_processor(model_path)
        fix_qwen2_vl_preprocessor(model_path)
        self.config = mlx_vlm.utils.load_config(
            model_path, trust_remote_code=trust_remote_code
        )
        self.trust_remote_code = trust_remote_code
        self.vocab_only = vocab_only
        self.model_path = model_path
        self._initializer()

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

        # Set the eos_token_ids
        eos_token_ids = None
        if (eos_tokens := self.config.get("eos_token_id", None)) is not None:
            if isinstance(eos_tokens, int):
                eos_token_ids = [eos_tokens]
            else:
                eos_token_ids = list(set(eos_tokens))
            logger.info(f"Setting eos token ids: {eos_token_ids}")

        # Use the mlx_lm tokenizer since it's more robust
        self.tokenizer = mlx_lm.tokenizer_utils.load(
            self.model_path, eos_token_ids=eos_token_ids
        )
        self.detokenizer = self.tokenizer.detokenizer

        self.cache_wrapper = None
        mx.clear_cache()

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
        images_b64: list[str],
        prompt_progress_callback,
        generate_args,
        max_image_size: tuple[int, int] | None,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Call this before starting evaluation

        This method opens the image from the base64-encoded string, and adds the special image token to the prompt

        Returns the processed prompt tokens to be input to the `generate_step` function, and optionally input
        embeddings. For VisionModelKit, the input embeddings are always none.
        """
        if self.has_processed_prompt:
            self._reset_for_prediction()

        self.model.process_prompt_with_images(
            images_b64, prompt_tokens, self.processor, self.detokenizer, max_image_size
        )
        self.has_processed_prompt = True

        # The VLM input_ids shape is important, but mlx_lm expects a flattened array
        #   Send back a fake shape and input_ids, and save the real shape in `self.model.input_ids`
        if len(images_b64) == 0:
            # For text-only, enable mlx-lm prompt processing
            return self.model.input_ids.reshape(-1), None
        # Disable mlx-lm prompt processing by returning a fake input
        return mx.array([0]), mx.array([0])

    def is_cross_prompt_cache_active(self) -> bool:
        """VisionModelKit does not support cross prompt caching"""
        return False

    def record_token_to_cache(self, token: int) -> None:
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
