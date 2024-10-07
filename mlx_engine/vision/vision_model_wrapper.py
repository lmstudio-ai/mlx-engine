import mlx.core as mx
import numpy as np
import sys
import PIL.Image


class VisionModelWrapper:
    """
    Wrapper class for Vision Models support
    This wrapper class adapts vision models so that they can be slotted into the mlx_lm generation engine
    Defines `__getattr__` and `__setattr__` to allow the model properties to be set/get as if it were a text model

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
            return self.vision_model(
                self.input_ids, self.pixel_values, self.mask, **kwargs
            )
        else:
            return self.vision_model.language_model(*args, mask=self.mask, **kwargs)

    def process_prompt_with_image(
        self,
        image: PIL.Image,
        prompt_tokens: mx.array,
        processor,
        detokenizer,
        image_token_format: str,
    ):
        """
        This method generates the input_ids, pixel_values, and mask for the vision model
        Call this before starting evaluation
        """
        detokenizer.reset()
        [detokenizer.add_token(token) for token in prompt_tokens]
        detokenizer.finalize()
        prompt = detokenizer.text
        prompt = prompt.replace("[img-1]", image_token_format)
        image_token = image_token_format.strip()
        sys.stderr.write(f"Prompt dump: {prompt}\n")

        def _prepare_inputs():
            """
            Adapted from `mlx_vlm.utils.prepare_inputs`, with the following enhancements:
            - Support more image token formats than just "<image>"
            - Handle prompts without images
            """
            mask = None
            if self.image_processor is not None:
                if image_token not in prompt:
                    return mx.array([processor(prompt).input_ids]), None, None
                text_chunks = [
                    processor(chunk).input_ids for chunk in prompt.split(image_token)
                ]
                input_ids = mx.array(
                    [
                        text_chunks[0]
                        + [self.vision_model.config.image_token_index]
                        + text_chunks[1]
                    ]
                )
                pixel_values = self.image_processor.preprocess(images=[image])[0]
                pixel_values = mx.array(np.expand_dims(pixel_values, axis=0))
            else:
                inputs = processor(prompt, image, return_tensors="np")
                pixel_values = inputs.get("pixel_values", None)
                pixel_values = (
                    mx.array(inputs["pixel_values"])
                    if pixel_values is not None
                    else None
                )
                input_ids = mx.array(inputs["input_ids"])
                mask = mx.array(inputs["attention_mask"])
                if "image_sizes" in inputs:
                    return input_ids, pixel_values, inputs["image_sizes"]
            return input_ids, pixel_values, mask

        self.input_ids, self.pixel_values, self.mask = _prepare_inputs()

    @property
    def vision_model(self):
        return self._model_attrs["vision_model"]

    @property
    def language_model(self):
        return self.vision_model.language_model

    @property
    def image_processor(self):
        return self._model_attrs["image_processor"]
