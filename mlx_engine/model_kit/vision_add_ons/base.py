from abc import abstractmethod

import mlx.core as mx
from mlx import nn


class BaseVisionAddOn:
    """
    Base class that defines the interface for a VisionAddOn.
    """

    @abstractmethod
    def __init__(self):
        """
        Where load of vision model components is intended to occur.
        """

    @abstractmethod
    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns input ids and input embeddings for the language model after text/image merging of the prompt.

        Args:
            text_model: Text model for embedding tokens
            prompt_tokens: Input prompt tokens
            images_b64: List of base64-encoded images
            max_size: Maximum image size as (width, height) tuple. If None, no resizing.
        """
