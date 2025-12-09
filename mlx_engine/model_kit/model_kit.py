import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from mlx_engine.cache_wrapper import (
    BranchingCacheWrapper,
    CacheWrapper,
)


# Import patch management
def apply_patches_by_config(config_path: Path) -> None:
    """
    Apply compatibility patches based on model configuration.

    Args:
        config_path: Path to the model's config.json file
    """
    try:
        import json

        with open(config_path, "r") as f:
            config = json.load(f)

        model_type = config.get("model_type")
        if not model_type:
            return

        # Apply ERNIE patches
        if model_type in ["ernie_4_5", "ernie_4_5_moe"]:
            try:
                from .patches.ernie_4_5 import apply_patches as apply_ernie_patches

                apply_ernie_patches()
                logger.info(f"Applied ERNIE patches for model type: {model_type}")
            except ImportError:
                logger.debug(
                    f"ERNIE patches not available for model type: {model_type}"
                )

        # Apply Gemma3n patches
        elif model_type == "gemma3n":
            try:
                from .patches.gemma3n import apply_patches as apply_gemma3n_patches

                apply_gemma3n_patches()
                logger.info(f"Applied Gemma3n patches for model type: {model_type}")
            except ImportError:
                logger.debug(
                    f"Gemma3n patches not available for model type: {model_type}"
                )

    except Exception as e:
        logger.error(f"Failed to apply patches for {config_path}: {e}")
        # Don't raise - patches should be transparent and not break model loading


from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.gemma3 import Gemma3VisionAddOn
from mlx_engine.model_kit.vision_add_ons.gemma3n import Gemma3nVisionAddOn
from mlx_engine.model_kit.vision_add_ons.lfm2_vl import LFM2VisionAddOn
from mlx_engine.model_kit.vision_add_ons.mistral3 import Mistral3VisionAddOn
from mlx_engine.model_kit.vision_add_ons.pixtral import PixtralVisionAddOn
from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_engine.utils.prompt_processing import process_prompt_text_only

logger = logging.getLogger(__name__)


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

    VISION_ADD_ON_MAP = {
        "gemma3": Gemma3VisionAddOn,
        "gemma3n": Gemma3nVisionAddOn,
        "lfm2-vl": LFM2VisionAddOn,
        "mistral3": Mistral3VisionAddOn,
        "pixtral": PixtralVisionAddOn,
        # qwen vl ports are bugged: https://github.com/lmstudio-ai/mlx-engine/issues/237
        # "qwen2_vl": Qwen2_VLVisionAddOn,
        # "qwen2_5_vl": Qwen2_VLVisionAddOn,
        # "qwen3_vl_moe": Qwen3_VL_MoEVisionAddOn,
        # "qwen3_vl": Qwen3_VLVisionAddOn,
    }

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    _cross_prompt_cache_active: bool = False
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    draft_model: Optional[nn.Module] = None
    model_type: Optional[str] = None

    # multi-modal add-ons
    vision_add_on: Optional[BaseVisionAddOn] = None

    # branching cache support
    branching_cache: Optional[BranchingCacheWrapper] = None
    enable_branching: bool = False

    def _vocab_only_init(self, model_path: Path):
        logger.info(f"Loading model (vocab-only) from {model_path}...")
        # Apply compatibility patches before loading tokenizer
        apply_patches_by_config(model_path / "config.json")
        self.tokenizer = mlx_lm.tokenizer_utils.load(model_path)
        self.detokenizer = self.tokenizer.detokenizer
        logger.info("Model (vocab-only) loaded successfully")

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
            logger.warning("max_kv_size is ignored when using KV cache quantization")
            max_kv_size = None
        self.model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        config_json = json.loads((model_path / "config.json").read_text())
        self.model_type = config_json.get("model_type", None)

        # Apply compatibility patches before loading the model
        apply_patches_by_config(model_path / "config.json")

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
        vision_add_on_class = self.VISION_ADD_ON_MAP.get(self.model_type)
        should_load_vision_add_on = (
            vision_add_on_class is not None and "vision_config" in config_json
        )
        if should_load_vision_add_on:
            self.vision_add_on = vision_add_on_class(model_path)
        logger.info("Model loaded successfully")

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
        prompt_progress_callback: Optional[Callable[[float], Union[bool, None]]],
        generate_args: dict,
        max_image_size: tuple[int, int] | None,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        ### TEXT-ONLY PROCESS_PROMPT ###
        is_text_only_processing = images_b64 is None or len(images_b64) == 0
        if is_text_only_processing:
            self._cross_prompt_cache_active = True
            if len(prompt_tokens) == 0:
                logger.warning(
                    "Received empty prompt. Generation quality will likely be poor"
                )
                # Models expect some sort of input, so add whitespace
                prompt_tokens = self.tokenize(" ")
            return process_prompt_text_only(
                mx.array(prompt_tokens),
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
        self._cross_prompt_cache_active = False
        input_ids, embeddings = self.vision_add_on.compute_embeddings(
            self.model, prompt_tokens, images_b64, max_size=max_image_size
        )
        return input_ids, embeddings

    def is_cross_prompt_cache_active(self) -> bool:
        """
        Check if cross-prompt caching is currently enabled.
        Can be overridden by subclasses for custom behavior.
        """
        return self._cross_prompt_cache_active

    def record_token_to_cache(self, token: int) -> None:
        self.cache_wrapper.record_generated_token(token)

    @staticmethod
    def is_supported_vision_arch(model_arch: str) -> bool:
        """
        Determines if the specified model architecture has vision support.

        Args:
            model_arch (str): The model architecture identifier to check

        Returns:
            bool: True if vision is supported, False otherwise
        """
        return model_arch in ModelKit.VISION_ADD_ON_MAP

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        path = Path(path)
        if self.tokenizer is None:
            logger.warning(
                "Draft model compatibility check requires at least a vocab-only loaded main model"
            )
            return False
        if self.vision_add_on is not None:
            logger.warning("Draft models are currently unsupported for vision models")
            return False
        draft_tokenizer = mlx_lm.tokenizer_utils.load(path)
        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            return False
        return True

    def load_draft_model(self, path: str | Path) -> None:
        logger.info(f"Loading draft model from {path}...")
        path = Path(path)
        if self.model is None:
            raise ValueError("Main model must be loaded before loading a draft model")
        if not self.is_draft_model_compatible(path):
            raise ValueError("Draft model is not compatible with main model")
        self.draft_model, _ = mlx_lm.utils.load(path)
        self.cache_wrapper.set_draft_model(self.draft_model)
        logger.info("Draft model loaded")

    def unload_draft_model(self) -> None:
        if self.draft_model is None:
            logger.info("No loaded draft model to unload")
        else:
            self.draft_model = None
            self.cache_wrapper.unset_draft_model()
        # Noticed that draft model memory would not be released without clearing metal cache
        mx.clear_cache()

    def enable_branching_cache(
        self,
        max_slots: int = 4,
        eviction_policy: str = "lru",
        memory_headroom_ratio: float = 0.1,
    ) -> None:
        """
        Enable branching cache support for this model kit.

        Args:
            max_slots: Maximum number of cache slots to maintain
            eviction_policy: Eviction policy ('lru' currently supported)
            memory_headroom_ratio: Ratio of memory to keep free (0.0-1.0)
        """
        if self.enable_branching and self.branching_cache is not None:
            logger.info("Branching cache already enabled")
            return

        self.branching_cache = BranchingCacheWrapper(
            max_slots=max_slots,
            eviction_policy=eviction_policy,
            memory_headroom_ratio=memory_headroom_ratio,
        )
        self.enable_branching = True
        logger.info(f"Enabled branching cache with max_slots={max_slots}")

    def checkpoint_branch(
        self, branch_id: str, prompt_hash: Optional[str] = None, pin: bool = False
    ) -> None:
        """
        Checkpoint the current cache state for a branch.

        Args:
            branch_id: Unique identifier for the branch
            prompt_hash: Optional hash of the prompt for identification
            pin: Whether to pin this branch to prevent eviction

        Raises:
            RuntimeError: If branching cache is not enabled
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )
        if self.cache_wrapper is None:
            raise RuntimeError("No cache wrapper available for checkpointing")

        self.branching_cache.checkpoint_branch(
            branch_id=branch_id,
            cache=self.cache_wrapper.cache,
            prompt_hash=prompt_hash,
            pin=pin,
        )

    def restore_branch(self, branch_id: str) -> None:
        """
        Restore a cached branch state.

        Args:
            branch_id: The branch identifier to restore

        Raises:
            RuntimeError: If branching cache is not enabled
            KeyError: If the branch is not found
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )
        if self.cache_wrapper is None:
            raise RuntimeError("No cache wrapper available for restoration")

        restored_cache = self.branching_cache.restore_branch(branch_id)
        self.cache_wrapper.cache = restored_cache

    def release_branch(self, branch_id: str) -> None:
        """
        Release a branch from the cache.

        Args:
            branch_id: The branch identifier to release

        Raises:
            RuntimeError: If branching cache is not enabled
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        self.branching_cache.release_branch(branch_id)

    def pin_branch(self, branch_id: str) -> None:
        """
        Pin a branch to prevent eviction.

        Args:
            branch_id: The branch identifier to pin

        Raises:
            RuntimeError: If branching cache is not enabled
            KeyError: If the branch is not found
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        self.branching_cache.pin_branch(branch_id)

    def unpin_branch(self, branch_id: str) -> None:
        """
        Unpin a branch to allow eviction.

        Args:
            branch_id: The branch identifier to unpin

        Raises:
            RuntimeError: If branching cache is not enabled
            KeyError: If the branch is not found
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        self.branching_cache.unpin_branch(branch_id)

    def get_cache_stats(self) -> dict:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache performance metrics

        Raises:
            RuntimeError: If branching cache is not enabled
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        return self.branching_cache.get_cache_stats()

    def list_branches(self) -> List[str]:
        """
        Get a list of all cached branch IDs.

        Returns:
            List of branch identifiers

        Raises:
            RuntimeError: If branching cache is not enabled
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        return self.branching_cache.list_branches()

    def get_branch_info(self, branch_id: str) -> Optional[dict]:
        """
        Get detailed information about a specific branch.

        Args:
            branch_id: The branch identifier

        Returns:
            Dictionary with branch information, or None if not found

        Raises:
            RuntimeError: If branching cache is not enabled
        """
        if not self.enable_branching or self.branching_cache is None:
            raise RuntimeError(
                "Branching cache not enabled. Call enable_branching_cache() first."
            )

        return self.branching_cache.get_branch_info(branch_id)
