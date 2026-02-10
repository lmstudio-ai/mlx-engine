import mlx.core as mx
import logging

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.base import InputEmbeddingsFeatures
from typing import List, Optional
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)

logger = logging.getLogger(__name__)


class VisionModelWrapper:
    """
    Wrapper class for Vision Models support
    This wrapper class adapts mlx-vlm models so that they can be slotted into the mlx_lm generation engine
    This wrapper defines `__getattr__` and `__setattr__` to allow the model properties to be set/get as if it were a text model

    Models are evaluated in `mlx_lm` with the `__call__` method, so define a custom `__call__` method to forward calls to the vision model
    """

    def __init__(self, model):
        """
        Set the class members in this unusual way, so that we can define `__getattr__` and `__setattr__`
        """
        self.__dict__["_model_attrs"] = {
            "vision_model": model,
            "input_ids": None,
            "pixel_values": None,
            "mask": None,
            "first_call": False,
            "decoder_input_ids": None,
            "language_model_kwargs": {},
            # vision model kwargs
            "model_inputs": {},
        }

    def __getattr__(self, name):
        """
        First, check if the name is a member of this class
        Then, check if the name is a member of the language model
        Finally, check if the name is a member of the vision model
        """
        if name in self._model_attrs:
            return self._model_attrs[name]
        try:
            return getattr(self.vision_model.language_model, name)
        except AttributeError:
            pass
        return getattr(self.vision_model, name)

    def __setattr__(self, name, value):
        """
        Set attribute of this class if it's not a member of the vision model
        """
        if name in self._model_attrs or not hasattr(self.vision_model, name):
            self._model_attrs[name] = value
        else:
            setattr(self.vision_model, name, value)

    def __call__(self, *args, input_embeddings=None, **kwargs):
        """
        This mirrors mlx-vlm's native generation loop (`mlx_vlm.generate.generate_step`):
        do one multimodal prompt/prefill step (via `get_input_embeddings`) and then
        forward all subsequent single-token decoding calls directly to the language model.
        ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L229
        """
        if self.pixel_values is not None and not self.first_call:
            self.first_call = True

            # Create a cache for the underlying language model. We do this explicitly
            # because `mlx_lm` creates caches based on the wrapper object, but mlx-vlm
            # language models often need custom cache types/layers.
            if hasattr(self.language_model, "make_cache"):
                cache = self.language_model.make_cache()
            else:
                kv_heads = (
                    [self.language_model.n_kv_heads] * len(self.language_model.layers)
                    if isinstance(self.language_model.n_kv_heads, int)
                    else self.language_model.n_kv_heads
                )
                cache = [KVCache() for _ in kv_heads]

            # Replace the mlx_lm cache with the one we created
            kwargs["cache"] = cache

            embedding_output = self.vision_model.get_input_embeddings(
                input_ids=self.input_ids,
                pixel_values=self.pixel_values,
                mask=self.mask,
                **self.model_inputs,
            )

            # Expect InputEmbeddingsFeatures here to match mlx-vlm's `generate_step` flow
            # https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L383-L396
            if not isinstance(embedding_output, InputEmbeddingsFeatures):
                raise TypeError(
                    "vision_model.get_input_embeddings(...) must return InputEmbeddingsFeatures. "
                    f"Got {type(embedding_output)}."
                )

            inputs_embeds = embedding_output.inputs_embeds
            if inputs_embeds is None:
                raise ValueError(
                    "vision_model.get_input_embeddings(...) returned InputEmbeddingsFeatures "
                    "without `inputs_embeds`."
                )

            lm_call_kwargs = {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }

            # Mirror model.__call__ behavior for models that produce a 4D attention mask.
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/models/gemma3/gemma3.py#L186-L192
            attention_mask_4d = lm_call_kwargs.pop("attention_mask_4d", None)
            if attention_mask_4d is not None:
                lm_call_kwargs["mask"] = attention_mask_4d

            outputs = self.language_model(
                self.input_ids,
                inputs_embeds=inputs_embeds,
                cache=cache,
                **lm_call_kwargs,
            )

            # Persist only decode type kwargs + cache to mirror mlx-vlm's native generation loop
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L369-L377
            persisted_kwargs = {"cache": cache}
            if outputs.cross_attention_states is not None:
                persisted_kwargs["cross_attention_states"] = (
                    outputs.cross_attention_states
                )
            elif outputs.encoder_outputs is not None:
                # `decoder_input_ids` is updated each step via `record_sampled_token`.
                self.decoder_input_ids = self.input_ids
                persisted_kwargs["decoder_input_ids"] = self.decoder_input_ids
                persisted_kwargs["encoder_outputs"] = outputs.encoder_outputs

            self.language_model_kwargs = persisted_kwargs
        else:
            try:
                # This is only missing if self.pixel_values is None
                if "cache" in self.language_model_kwargs:
                    # Use the cache from self.language_model_kwargs
                    kwargs.pop("cache", None)

                lm_call_kwargs = dict(self.language_model_kwargs)

                # Mirrors mlx-vlm's `generate_step` continuation path for encoder-decoder models:
                # https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L332-L336
                if "decoder_input_ids" in lm_call_kwargs:
                    # Avoid passing decoder_inputs_embeds alongside decoder_input_ids.
                    lm_call_kwargs.pop("decoder_inputs_embeds", None)
                    lm_call_kwargs["decoder_input_ids"] = self.decoder_input_ids
                    outputs = self.language_model(
                        **kwargs,
                        **lm_call_kwargs,
                    )
                else:
                    outputs = self.language_model(
                        *args,
                        **kwargs,
                        **lm_call_kwargs,
                    )

            except ValueError as e:
                # Create a friendly error message if a user tries to use mllama without images
                if "Cross attention states must be provided for layer" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e

        return outputs.logits

    def record_sampled_token(self, token: int) -> None:
        # use record_sampled_token as the mechanism to update decoder_input_ids
        # properly for each step of generation.
        # Native mlx-vlm does this here: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L372-L375
        self.decoder_input_ids = mx.array([[token]])

    def process_prompt_with_images(
        self,
        images_b64: Optional[List[str]],
        prompt_tokens: mx.array,
        processor,
        detokenizer,
        max_image_size: tuple[int, int] | None,
    ):
        """
        This method generates the input_ids, pixel_values, and mask for the vision model
        Call this before starting evaluation
        """
        if images_b64 is None:
            images_b64 = []

        # Handle the case with no images
        if len(images_b64) == 0:
            detokenizer.reset()
            [detokenizer.add_token(token) for token in prompt_tokens]
            detokenizer.finalize()
            prompt = detokenizer.text

            logger.debug(f"Prompt dump: {prompt}\n")

            try:
                if hasattr(processor, "process"):
                    # Needed for Molmo
                    self.input_ids = mx.array(
                        processor.process(text=prompt)["input_ids"]
                    )
                else:
                    self.input_ids = mx.array(processor(text=prompt).input_ids)
            except ValueError as e:
                if "`images` are expected as arguments" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e
        else:
            # Use the common function for image processing
            processed = common_process_prompt_with_images(
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                processor=processor,
                config=self.vision_model.config,
                max_size=max_image_size,
            )

            # Set class attributes from the processed result
            self.input_ids = processed.input_ids
            self.pixel_values = processed.pixel_values
            self.mask = processed.attention_mask
            self.model_inputs = processed.other_inputs

    @property
    def vision_model(self):
        return self._model_attrs["vision_model"]

    @property
    def language_model(self):
        return self.vision_model.language_model
