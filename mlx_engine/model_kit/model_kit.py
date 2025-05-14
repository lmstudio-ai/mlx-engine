import json
from typing import Optional, List, Tuple

import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer

from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx

from mlx_engine.logging import log_info, log_warn
from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.gemma3 import Gemma3VisionAddOn
from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_engine.utils.prompt_processing import process_prompt_text_only

LOG_PREFIX = "ModelKit"


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a model.

    Args:
        model_path (Path): Path to the model directory containing model files.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        kv_bits (Optional[int]): Number of bits for KV cache quantization. None disables quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization. Defaults to 64.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled. Defaults to 0.
    """

    SUPPORTED_VISION_ARCHITECTURES = ["gemma3"]
    VISION_ADD_ON_MAP = {
        "gemma3": Gemma3VisionAddOn,
    }

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    draft_model: Optional[nn.Module] = None

    # multi-modal add-ons
    vision_add_on: Optional[BaseVisionAddOn] = None

    def _vocab_only_init(self, model_path: Path):
        log_info(
            prefix=LOG_PREFIX,
            message=f"Loading model (vocab-only) from {model_path}...",
        )
        self.tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(model_path)
        self.detokenizer = self.tokenizer.detokenizer
        log_info(prefix=LOG_PREFIX, message="Model (vocab-only) loaded successfully")

    def _full_model_init(
        self,
        model_path: Path,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )
        if kv_bits and max_kv_size is not None:
            # Quantized KV cache is only supported for non-rotating KV cache
            log_warn(
                prefix=LOG_PREFIX,
                message="max_kv_size is ignored when using KV cache quantization",
            )
            max_kv_size = None
        self.model_path = model_path
        log_info(prefix=LOG_PREFIX, message=f"Loading model from {model_path}...")
        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(
            self.model,
            max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        config_json = json.loads((model_path / "config.json").read_text())
        model_type = config_json.get("model_type", None)
        vision_add_on_class = self.VISION_ADD_ON_MAP.get(model_type)
        if vision_add_on_class:
            self.vision_add_on = vision_add_on_class(model_path)
        log_info(prefix=LOG_PREFIX, message="Model loaded successfully")

    def __init__(
        self,
        model_path: Path,
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        if vocab_only:
            self._vocab_only_init(model_path)
        else:
            self._full_model_init(
                model_path,
                max_kv_size,
                kv_bits,
                kv_group_size,
                quantized_kv_start,
            )

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_callback,
        generate_args,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        ### TEXT-ONLY PROCESS_PROMPT ###
        is_text_only_processing = images_b64 is None or len(images_b64) == 0
        if is_text_only_processing:
            return process_prompt_text_only(
                prompt_tokens,
                self.cache_wrapper,
                generate_args,
                self.draft_model,
                speculative_decoding_toggle,
                prompt_progress_callback,
            ), None
        ### WITH IMAGES PROMPT PROCESSING ###s
        if self.vision_add_on is None:
            raise ValueError(
                "Vision add-on is not loaded, but images were provided for processing"
            )
        embeddings = self.vision_add_on.compute_embeddings(
            self.model, prompt_tokens, images_b64
        )
        return [], embeddings

    def record_token_to_cache(self, token: int) -> None:
        self.cache_wrapper.record_generated_token(token)
        pass

    @staticmethod
    def is_supported_vision_arch(model_arch: str) -> bool:
        """
        Determines if the specified model architecture has vision support.

        Args:
            model_arch (str): The model architecture identifier to check

        Returns:
            bool: True if vision is supported, False otherwise
        """
        return model_arch in ModelKit.SUPPORTED_VISION_ARCHITECTURES

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        path = Path(path)
        if self.tokenizer is None:
            log_warn(
                prefix=LOG_PREFIX,
                message="Draft model compatibility check requires at least a vocab-only "
                "loaded main model",
            )
            return False
        if self.vision_add_on is not None:
            log_warn(
                prefix=LOG_PREFIX,
                message="Draft models are currently unsupported for vision models",
            )
            return False
        draft_tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(path)
        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            return False
        return True

    def load_draft_model(self, path: str | Path) -> None:
        log_info(prefix=LOG_PREFIX, message=f"Loading draft model from {path}...")
        path = Path(path)
        if self.model is None:
            raise ValueError("Main model must be loaded before loading a draft model")
        if not self.is_draft_model_compatible(path):
            raise ValueError("Draft model is not compatible with main model")
        self.draft_model, _ = mlx_lm.utils.load(path)
        self.cache_wrapper.set_draft_model(self.draft_model)
        log_info(prefix=LOG_PREFIX, message="Draft model loaded")

    def unload_draft_model(self) -> None:
        if self.draft_model is None:
            log_info(prefix=LOG_PREFIX, message="No loaded draft model to unload")
        else:
            self.draft_model = None
            self.cache_wrapper.unset_draft_model()
        # Noticed that draft model memory would not be released without clearing metal cache
        mx.clear_cache()
