import mlx.core as mx
import logging

from mlx_vlm.models.cache import KVCache, SimpleKVCache
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
        See this reference implementation
        https://github.com/Blaizzy/mlx-vlm/blob/6c98971/mlx_vlm/utils.py#L783-L810

        In the reference implementation, the vision model is called once at the beginning,
        then all subsequent calls are forwarded to the language model. Mirror that behavior here.
        """
        if self.pixel_values is not None and not self.first_call:
            self.first_call = True

            # taken from here https://github.com/Blaizzy/mlx-vlm/blob/2974401/mlx_vlm/utils.py#L987
            if hasattr(self.language_model, "make_cache"):
                cache = self.language_model.make_cache()
            else:
                kv_heads = (
                    [self.language_model.n_kv_heads] * len(self.language_model.layers)
                    if isinstance(self.language_model.n_kv_heads, int)
                    else self.language_model.n_kv_heads
                )
                if self.vision_model.config.model_type == "florence2":
                    cache = [
                        (SimpleKVCache(), SimpleKVCache())
                        for n in self.language_model.layers
                    ]
                else:
                    cache = [KVCache() for n in kv_heads]

            # Replace the mlx_lm cache with the one we created
            kwargs["cache"] = cache

            outputs = self.vision_model(
                self.input_ids,
                self.pixel_values,
                mask=self.mask,
                **self.model_inputs,
                **kwargs,
            )

            # taken from here https://github.com/Blaizzy/mlx-vlm/blob/2974401/mlx_vlm/utils.py#L1045-L1056
            if outputs.cross_attention_states is not None:
                self.language_model_kwargs = {
                    k: v
                    for k, v in zip(
                        ["cross_attention_states"], [outputs.cross_attention_states]
                    )
                }
            elif outputs.encoder_outputs is not None:
                self.decoder_input_ids = self.input_ids
                self.language_model_kwargs = {
                    "decoder_input_ids": self.decoder_input_ids,
                    "encoder_outputs": outputs.encoder_outputs,
                }
            # elif self.vision_model.config.model_type == "qwen3_vl":
            #     self.language_model_kwargs = {
            #         "visual_pos_masks": outputs.visual_pos_masks,
            #         "deepstack_visual_embeds": outputs.deepstack_visual_embeds,
            #     }

            # Add the cache we created here to the language model kwargs
            self.language_model_kwargs["cache"] = cache
        else:
            try:
                if (
                    "cache" in self.language_model_kwargs
                ):  # This is only missing if self.pixel_values is None
                    del kwargs["cache"]  # Use the cache from self.language_model_kwargs

                # taken from here https://github.com/Blaizzy/mlx-vlm/blob/2974401/mlx_vlm/utils.py#L1009
                if "decoder_input_ids" in self.language_model_kwargs:
                    self.language_model_kwargs["decoder_input_ids"] = (
                        self.decoder_input_ids
                    )
                    outputs = self.language_model(
                        **kwargs,
                        **self.language_model_kwargs,
                    )
                else:
                    outputs = self.language_model(
                        *args,
                        **kwargs,
                        **self.language_model_kwargs,
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
        # Adapted from here https://github.com/Blaizzy/mlx-vlm/blob/2974401/mlx_vlm/utils.py#L1064
        self.decoder_input_ids = mx.array([token])

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
