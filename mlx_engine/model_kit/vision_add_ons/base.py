from abc import abstractmethod
from typing import List, Tuple

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
        images_b64: List[str],
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns input embeddings for the language model after text/image merging of the prompt
        """
