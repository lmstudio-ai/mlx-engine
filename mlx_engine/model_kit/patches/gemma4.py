"""
Patch Gemma 4 so unified multimodal prompts reuse the full prompt's masked
per-layer-input token ids during chunked prefill.
"""

from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models.gemma4_text import Gemma4TextModel

# Stable alias to the pristine mlx-lm class captured before apply_patches()
# mutates mlx_lm.models.gemma4_text in place.
OriginalGemma4TextModel = Gemma4TextModel


class PatchedGemma4TextModel(OriginalGemma4TextModel):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_per_layer_input_ids = None

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        if (
            per_layer_inputs is None
            and input_embeddings is not None
            and self.prompt_per_layer_input_ids is not None
        ):
            prompt_per_layer_input_ids = self.prompt_per_layer_input_ids
            if prompt_per_layer_input_ids.shape[1] != input_embeddings.shape[-2]:
                start = self._cache_offset(cache)
                target_len = input_embeddings.shape[-2]
                prompt_per_layer_input_ids = prompt_per_layer_input_ids[
                    :, start : start + target_len
                ]
            per_layer_inputs = self._get_per_layer_inputs(prompt_per_layer_input_ids)

        return super().__call__(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    @staticmethod
    def _cache_offset(cache: Optional[Any]) -> int:
        for layer_cache in cache or []:
            offset = getattr(layer_cache, "offset", None)
            if offset is None:
                continue
            if isinstance(offset, int):
                return offset
            if isinstance(offset, mx.array) and offset.ndim == 0:
                return offset.item()
            if isinstance(offset, mx.array):
                return offset[0].item()
        return 0


def apply_patches():
    import mlx_lm.models.gemma4_text

    mlx_lm.models.gemma4_text.Gemma4TextModel = PatchedGemma4TextModel
