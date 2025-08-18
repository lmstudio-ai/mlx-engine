from pathlib import Path

from mlx import nn
import mlx.core as mx

import mlx_vlm.utils

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.utils.image_utils import convert_to_pil
from mlx_engine.logging import log_info


class LanguageModelWrapper(nn.Module):
    """Wrapper to make mlx-vlm's LanguageModel compatible with mlx_lm interface"""

    def __init__(self, language_model):
        super().__init__()
        object.__setattr__(self, "_wrapped_model", language_model)

    def __call__(self, inputs, input_embeddings=None, mask=None, cache=None, **kwargs):
        # Map input_embeddings to inputs_embeds for mlx-vlm compatibility
        output = self._wrapped_model(
            inputs=inputs,
            inputs_embeds=input_embeddings,
            mask=mask,
            cache=cache,
            **kwargs,
        )

        # mlx-vlm returns LanguageModelOutput, but mlx_lm expects raw logits
        return output.logits

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped language model
        if hasattr(self, "_wrapped_model"):
            return getattr(self._wrapped_model, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


class Qwen2_5_VLVisionAddOn(BaseVisionAddOn):
    """
    Unified vision add-on for Qwen2-VL and Qwen2.5-VL models. Leverages the models' unified
    vision-language architecture by loading the complete model via mlx-vlm and wrapping the
    language component for mlx-lm compatibility.
    """

    Qwen2_5_VL_LOG_PREFIX = "Qwen2_5_VLVisionAddOn"

    def __init__(self, model_path: Path):
        """Initialize QwenVLVisionAddOn with components loaded from the given path."""
        super().__init__()

        # Load the complete model using mlx-vlm. It will automatically instantiate the correct
        # model type (qwen2_vl.Model or qwen2_5_vl.Model) based on config.json
        full_model, processor = mlx_vlm.utils.load(
            model_path,
            processor_config={"trust_remote_code": True},
            trust_remote_code=True,
        )

        # Wrap the language model to be compatible with mlx_lm interface
        self.language_model = LanguageModelWrapper(full_model.language_model)

        # Extract components from the loaded model
        self.config = full_model.config
        self.processor = processor

        # Keep reference to full model for get_input_embeddings
        self.full_model = full_model

        log_info(
            prefix=self.Qwen2_5_VL_LOG_PREFIX,
            message=f"Vision add-on loaded successfully for {model_path} (model_type: {self.config.model_type})",
        )

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
    ) -> tuple[mx.array, mx.array]:
        """
        Compute input_ids and embeddings for text with images. For Qwen2/2.5-VL models, we bypass
        the `common_process_prompt_with_images` function because padding images to the same size
        causes issues with multiple image handling. The model's processor handles different-sized
        images correctly when not padded.
        """

        # Convert base64 images to PIL format
        images = convert_to_pil(images_b64)

        # Detokenize to get prompt text
        detokenizer = self.processor.detokenizer
        detokenizer.reset()
        [detokenizer.add_token(token) for token in prompt_tokens]
        detokenizer.finalize()
        prompt = detokenizer.text

        # Get image token index for the model
        image_token_index = self.config.image_token_id

        # Process inputs using mlx-vlm's `prepare_inputs` without resizing
        # Qwen2/2.5-VL can handle variable-sized images natively, avoiding issues with padding
        inputs = mlx_vlm.utils.prepare_inputs(
            processor=self.processor,
            images=images,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=None,  # Preserve native dimensions
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Use the full model's get_input_embeddings method to merge image features
        # This method handles model-specific differences internally
        final_inputs_embeds = self.full_model.get_input_embeddings(
            input_ids, pixel_values, image_grid_thw
        )

        # Remove batch dimension for compatibility with mlx_lm
        return input_ids.squeeze(0), final_inputs_embeds.squeeze(0)
