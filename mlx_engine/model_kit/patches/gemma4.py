"""
Gemma 4 compatibility patch for unified multimodal prompts.

The upstream mlx-lm Gemma4TextModel reconstructs per-layer inputs from the
current prompt chunk when the caller does not pass them explicitly. That is
fine for text-only prompts, but incorrect for unified multimodal prompts
because Gemma 4 image/audio token ids fall inside the per-layer-input vocab
and must be masked to 0 before lookup, matching mlx-vlm.

This patch lets mlx-engine stash the full prompt's masked per-layer inputs on
the text model and have later prefill chunks slice the correct window from
that stored state.
"""

from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models.gemma4_text import Gemma4TextModel

OriginalGemma4TextModel = Gemma4TextModel


class PatchedGemma4TextModel(Gemma4TextModel):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_per_layer_inputs = None

    def reset_prompt_per_layer_input_state(self) -> None:
        self.prompt_per_layer_inputs = None

    def set_prompt_per_layer_inputs(
        self,
        prompt_per_layer_inputs: Optional[mx.array],
    ) -> None:
        if prompt_per_layer_inputs is None:
            self.prompt_per_layer_inputs = None
            return
        if prompt_per_layer_inputs.ndim == 3:
            prompt_per_layer_inputs = prompt_per_layer_inputs[None]
        self.prompt_per_layer_inputs = prompt_per_layer_inputs

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        effective_per_layer_inputs = per_layer_inputs
        if (
            effective_per_layer_inputs is None
            and input_embeddings is not None
            and self.prompt_per_layer_inputs is not None
        ):
            effective_per_layer_inputs = self.prompt_per_layer_inputs

        if effective_per_layer_inputs is not None:
            effective_per_layer_inputs = self._slice_per_layer_inputs(
                per_layer_inputs=effective_per_layer_inputs,
                cache=cache,
                batch_size=(
                    input_embeddings.shape[0]
                    if input_embeddings is not None
                    else inputs.shape[0]
                ),
                target_len=(
                    input_embeddings.shape[1]
                    if input_embeddings is not None
                    else inputs.shape[1]
                ),
            )

        return super().__call__(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=effective_per_layer_inputs,
        )

    def _slice_per_layer_inputs(
        self,
        *,
        per_layer_inputs: mx.array,
        cache: Optional[Any],
        batch_size: int,
        target_len: int,
    ) -> mx.array:
        if per_layer_inputs.ndim == 3:
            per_layer_inputs = per_layer_inputs[None]
        if per_layer_inputs.ndim != 4:
            raise ValueError(
                "Gemma 4 prompt per-layer inputs must have shape "
                "(batch, seq, num_layers, hidden)."
            )
        if per_layer_inputs.shape[0] != batch_size:
            if per_layer_inputs.shape[0] == 1:
                per_layer_inputs = mx.broadcast_to(
                    per_layer_inputs,
                    (batch_size,) + tuple(per_layer_inputs.shape[1:]),
                )
            else:
                raise ValueError(
                    "Gemma 4 prompt per-layer inputs batch dimension does not "
                    "match the current input batch size."
                )
        if per_layer_inputs.shape[1] < target_len:
            raise ValueError(
                "Gemma 4 prompt per-layer inputs are shorter than the current "
                "input chunk."
            )
        if per_layer_inputs.shape[1] == target_len:
            return per_layer_inputs

        cache_offset = self._cache_offset(cache)
        max_start = max(per_layer_inputs.shape[1] - target_len, 0)
        start = min(cache_offset, max_start)
        return per_layer_inputs[:, start : start + target_len]

    @staticmethod
    def _cache_offset(cache: Optional[Any]) -> int:
        for layer_cache in cache or []:
            if layer_cache is None or not hasattr(layer_cache, "offset"):
                continue
            offset = layer_cache.offset
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
