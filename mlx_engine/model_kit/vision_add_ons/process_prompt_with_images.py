from typing import List, Union, NamedTuple
import mlx.core as mx
from mlx_vlm import prepare_inputs
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import logging

from mlx_engine.utils.image_utils import convert_to_pil, custom_resize

logger = logging.getLogger(__name__)


class ProcessedImagePrompt(NamedTuple):
    input_ids: mx.array
    pixel_values: mx.array
    attention_mask: mx.array
    other_inputs: dict


def common_process_prompt_with_images(
    prompt_tokens: mx.array,
    images_b64: List[str],
    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    config,  # expected to be a ModelConfig object as defined by mlx-vlm. Can vary by model
) -> ProcessedImagePrompt:
    """
    Common prompt processing used by mlx-vlm vision add-ons.
    Returns a named tuple with all processed inputs.
    """
    if len(images_b64) == 0:
        raise ValueError("Images must be non-empty")
    detokenizer = processor.detokenizer
    detokenizer.reset()
    [detokenizer.add_token(token) for token in prompt_tokens]
    detokenizer.finalize()
    prompt = detokenizer.text

    logger.info(f"Prompt dump: {prompt}\n")

    images = convert_to_pil(images_b64)
    images = custom_resize(images)

    if hasattr(config, "image_token_index"):
        image_token_index = config.image_token_index
    elif hasattr(config.vision_config, "image_token_id"):
        image_token_index = config.vision_config.image_token_id
    else:
        image_token_index = None

    inputs = prepare_inputs(
        processor=processor,
        images=images,
        prompts=prompt,
        image_token_index=image_token_index,
        resize_shape=None,
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]
    other_model_inputs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    return ProcessedImagePrompt(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        other_inputs=other_model_inputs,
    )
