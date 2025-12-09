import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx_lm
import mlx_vlm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from mlx_engine.cache_wrapper import BranchingCacheWrapper
from mlx_engine.model_kit.model_kit import ModelKit

from ._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)
from .vision_model_wrapper import VisionModelWrapper

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
        images_b64: Optional[List[str]],
        prompt_progress_callback: Optional[Callable[[float], Union[bool, None]]],
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
        if images_b64 is None or len(images_b64) == 0:
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

    # branching cache support
    branching_cache: Optional[BranchingCacheWrapper] = None
    enable_branching: bool = False

    def enable_branching_cache(
        self,
        max_slots: int = 4,
        eviction_policy: str = "lru",
        memory_headroom_ratio: float = 0.1,
    ) -> None:
        """
        Enable branching cache support for this vision model kit.

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
        logger.info(
            f"Enabled branching cache for vision model with max_slots={max_slots}"
        )

    def checkpoint_branch(
        self, branch_id: str, prompt_hash: Optional[str] = None, pin: bool = False
    ) -> None:
        """
        Checkpoint the current model state for a branch.

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

        # For vision models, we checkpoint the entire model state
        self.branching_cache.checkpoint_branch(
            branch_id=branch_id, cache=self.model, prompt_hash=prompt_hash, pin=pin
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

        restored_model = self.branching_cache.restore_branch(branch_id)
        self.model = restored_model

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
