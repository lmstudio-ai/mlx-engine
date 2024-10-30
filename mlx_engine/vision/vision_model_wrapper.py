import mlx.core as mx
import numpy as np
import sys
from io import BytesIO
import base64
import PIL

from typing import List, Optional


class VisionModelWrapper:
    """
    Wrapper class for Vision Models support
    This wrapper class adapts vision models so that they can be slotted into the mlx_lm generation engine
    This wrapper defines `__getattr__` and `__setattr__` to allow the model properties to be set/get as if it were a text model

    Models are evaluated in `mlx_lm` with the `__call__` method, so define a custom `__call__` method to forward calls to the vision model
    """

    def __init__(self, model, image_processor):
        """
        Set the class members in this unusual way, so that we can define `__getattr__` and `__setattr__`
        """
        self.__dict__["_model_attrs"] = {
            "vision_model": model,
            "image_processor": image_processor,
            "input_ids": None,
            "pixel_values": None,
            "mask": None,
            "first_call": False,
            "language_model_kwargs": {},
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
            outputs = self.vision_model(
                self.input_ids,
                self.pixel_values,
                mask=self.mask,
                image_grid_thw=self.image_grid_thw,
                image_sizes=self.image_sizes,
                aspect_ratio_ids=self.aspect_ratio_ids,
                aspect_ratio_mask=self.aspect_ratio_mask,
                cross_attention_mask=self.cross_attention_mask,
                **kwargs,
            )

            # taken from here https://github.com/Blaizzy/mlx-vlm/blob/904023515d06543be28e3cfe4dd73fa2fe89cac9/mlx_vlm/utils.py#L916-L924
            if outputs.cross_attention_states is not None:
                self.language_model_kwargs = {
                    k: v
                    for k, v in zip(
                        ["cross_attention_states"], [outputs.cross_attention_states]
                    )
                }
        else:
            try:
                outputs = self.vision_model.language_model(
                    *args, mask=self.mask, **kwargs, **self.language_model_kwargs
                )
            except ValueError as e:
                # Create a friendly error message if a user tries to use mllama without images
                if "Cross attention states must be provided for layer" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e
        return outputs.logits

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
        detokenizer.reset()
        [detokenizer.add_token(token) for token in prompt_tokens]
        detokenizer.finalize()
        prompt = detokenizer.text

        sys.stderr.write(f"[mlx-engine] Prompt dump: {prompt}\n")

        (
            self.input_ids,
            self.pixel_values,
            self.mask,
            self.image_grid_thw,
            self.image_sizes,
            self.aspect_ratio_ids,
            self.aspect_ratio_mask,
            self.cross_attention_mask,
        ) = self._prepare_inputs(images_b64, prompt, processor)

    def _prepare_inputs(self, images_b64: list, prompt: str, processor):
        """
        Adapted from `mlx_vlm.utils.prepare_inputs`, with the following enhancements:
        - Handle prompts without images
        """
        pil_images = self._convert_to_pil(images_b64)
        resized_images = self._custom_resize(pil_images)

        mask = None
        image_grid_thw = None
        image_sizes = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        cross_attention_mask = None
        if len(images_b64) == 0:
            return (mx.array(processor(text=prompt).input_ids), *(None,) * 7)

        if self.image_processor is not None:
            processor.pad_token = processor.eos_token
            text_chunks = [
                [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            ]

            max_length = max(
                sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks
            )
            input_ids = []
            for chunks in text_chunks:
                ids = (
                        chunks[0] + [self.vision_model.config.image_token_index] + chunks[1]
                )
                padding = [processor.pad_token_id] * (max_length - len(ids))
                input_ids.append(mx.array(ids + padding))
            input_ids = mx.array(input_ids)
            pixel_values = self.image_processor.preprocess(images=resized_images)
            pixel_values = mx.array(np.stack(pixel_values))
            mask = mx.array(
                [(ids != processor.pad_token_id) for ids in input_ids]
            ).astype(mx.int32)
        else:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            inputs = processor(
                text=[prompt], images=resized_images, padding=True, return_tensors="mlx"
            )
            if isinstance(inputs["pixel_values"], list):
                pixel_values = inputs["pixel_values"]
            else:
                pixel_values = mx.array(inputs["pixel_values"])
            input_ids = mx.array(inputs["input_ids"])
            mask = mx.array(inputs["attention_mask"])
            image_sizes = inputs.get("image_sizes", None)
            if image_sizes is not None:
                image_sizes = mx.array(image_sizes)
            image_grid_thw = inputs.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = mx.array(image_grid_thw)
            aspect_ratio_ids = inputs.get("aspect_ratio_ids", None)
            if aspect_ratio_ids is not None:
                aspect_ratio_ids = mx.array(aspect_ratio_ids)
            aspect_ratio_mask = inputs.get("aspect_ratio_mask", None)
            if aspect_ratio_mask is not None:
                aspect_ratio_mask = mx.array(aspect_ratio_mask)
            cross_attention_mask = inputs.get("cross_attention_mask", None)
            if cross_attention_mask is not None:
                cross_attention_mask = mx.array(cross_attention_mask)

        return (
            input_ids,
            pixel_values,
            mask,
            image_grid_thw,
            image_sizes,
            aspect_ratio_ids,
            aspect_ratio_mask,
            cross_attention_mask,
        )

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

            sys.stderr.write(f"[mlx-engine] Image {i+1}: Original size {original_size}\n")

            if img.width > max_size[0] or img.height > max_size[1]:
                if img.width > img.height:
                    new_width = max_size[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_size[1]
                    new_width = int(new_height * aspect_ratio)
                img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
                sys.stderr.write(f"[mlx-engine] Image {i+1}: Resized to {img.width}x{img.height}\n")
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

    @property
    def image_processor(self):
        return self._model_attrs["image_processor"]
