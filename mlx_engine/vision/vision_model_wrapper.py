import mlx.core as mx
import numpy as np
import sys
from io import BytesIO
import base64
import PIL
from mlx_vlm.models.base import KVCache, SimpleKVCache
from mlx_vlm.utils import prepare_inputs

from typing import List, Optional


class VisionModelWrapper:
    """
    Wrapper class for Vision Models support
    This wrapper class adapts vision models so that they can be slotted into the mlx_lm generation engine
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

    def __call__(self, *args, **kwargs):
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
                    cache = [KVCache(self.language_model.head_dim, n) for n in kv_heads]

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
                        mask=self.mask,
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
    ):
        """
        This method generates the input_ids, pixel_values, and mask for the vision model
        Call this before starting evaluation
        """
        if images_b64 is None:
            images_b64 = []

        detokenizer.reset()
        [detokenizer.add_token(token) for token in prompt_tokens]
        detokenizer.finalize()
        prompt = detokenizer.text

        sys.stderr.write(f"[mlx-engine] Prompt dump: {prompt}\n")

        images = self._convert_to_pil(images_b64)
        images = self._custom_resize(images)

        if hasattr(self.vision_model.config, "image_token_index"):
            image_token_index = self.vision_model.config.image_token_index
        else:
            image_token_index = None

        if len(images) == 0:
            try:
                if hasattr(processor, "process"):
                    # Needed for Molmo
                    self.input_ids = mx.array(processor.process(text=prompt)["input_ids"])
                else:
                    self.input_ids = mx.array(processor(text=prompt).input_ids)
            except ValueError as e:
                if "`images` are expected as arguments" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e
        else:
            inputs = prepare_inputs(
                processor=processor,
                images=images,
                prompts=prompt,
                image_token_index=image_token_index,
                resize_shape=None
            )
            self.input_ids = inputs["input_ids"]
            self.pixel_values = inputs["pixel_values"]
            self.mask = inputs["attention_mask"]
            self.model_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }

    def _convert_to_pil(self, images_b64: List[str]):
        """Convert a list of base64 strings to PIL Images"""
        return [
            PIL.Image.open(BytesIO(base64.b64decode(img))) for img in images_b64 or []
        ]

    def _custom_resize(self, pil_images, max_size=(1000, 1000)):
        """
        Resize and optionally pad a list of PIL images.

        This function resizes images that exceed the specified maximum dimensions,
        maintaining their aspect ratios. If there is more than one image, it also
        pads all images to the same size.

        Args:
            pil_images (list): A list of PIL Image objects to be processed.
            max_size (tuple): Maximum allowed dimensions (width, height) for the images.
                            Defaults to (1000, 1000).

        Returns:
            list: A list of processed PIL Image objects. If there was only one input image,
                it returns the resized image without padding. If there were multiple input
                images, it returns padded images of uniform size.

        Side effects:
            Writes progress and status messages to sys.stderr.
        """
        resized_images = []
        max_width, max_height = 0, 0

        for i, img in enumerate(pil_images):
            original_size = (img.width, img.height)
            aspect_ratio = img.width / img.height

            sys.stderr.write(
                f"[mlx-engine] Image {i+1}: Original size {original_size}\n"
            )

            if img.width > max_size[0] or img.height > max_size[1]:
                if img.width > img.height:
                    new_width = max_size[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_size[1]
                    new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
                sys.stderr.write(
                    f"[mlx-engine] Image {i+1}: Resized to {img.width}x{img.height}\n"
                )
            else:
                sys.stderr.write(f"[mlx-engine] Image {i+1}: No resize needed\n")

            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)

            resized_images.append(img)

        if len(pil_images) > 1:
            sys.stderr.write(
                f"[mlx-engine] Maximum dimensions: {max_width}x{max_height}. "
                f"Adding padding so that all images are the same size.\n"
            )

            final_images = []
            for i, img in enumerate(resized_images):
                new_img = PIL.Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))
                paste_x = (max_width - img.width) // 2
                paste_y = (max_height - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                final_images.append(new_img)
            return final_images
        else:
            return resized_images

    @property
    def vision_model(self):
        return self._model_attrs["vision_model"]

    @property
    def language_model(self):
        return self.vision_model.language_model
