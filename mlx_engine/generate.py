from typing import Callable, Iterator, List, NamedTuple, Optional
import json
from pathlib import Path

import mlx_lm

from mlx_engine.model_kit import ModelKit
from mlx_engine.vision.vision_model_kit import VisionModelKit
from mlx_engine.processors.outlines_logits_processor import OutlinesLogitsProcessor
from mlx_engine.utils.top_logprobs import summarize_top_logprobs, TokenLogprob
from mlx_engine.stop_processor import StopProcessor, GenerationStopCondition
from mlx_engine.utils.set_seed import set_seed


MAX_TOP_LOGPROBS = 10


class GenerationResult(NamedTuple):
    text: str
    tokens: List[TokenLogprob]
    top_logprobs: List[List[TokenLogprob]]
    stop_condition: Optional[GenerationStopCondition]


def load_model(
    model_path: str | Path, max_kv_size: int, trust_remote_code: bool
) -> ModelKit | VisionModelKit:
    """
    Load a language model or vision-language model from the specified path.

    Args:
        model_path (str | Path): Path to the model directory containing model files and config.json
        max_kv_size (int): Maximum key-value cache size for the model
        trust_remote_code (bool): Whether to trust and load remote code during model initialization

    Returns:
        ModelKit | VisionModelKit: An instance of either ModelKit for language models or
            VisionModelKit for vision-language models, depending on the model type

    Raises:
        FileNotFoundError: If config.json is not found in the model path
        json.JSONDecodeError: If config.json is not valid JSON
    """
    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())

    if "vision_config" in config_json:
        return VisionModelKit(model_path, trust_remote_code)
    else:
        return ModelKit(model_path, max_kv_size)


# Adapted from mlx_lm.utils.stream_generate
def create_generator(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    prompt_progress_callback: Optional[Callable[[float], None]],
    images_b64: Optional[List[str]],
    stop_strings: Optional[List[str]],
    generate_args: dict,
    top_logprobs: Optional[int] = None,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    Args:
        model_kit (ModelKit | VisionModelKit): The model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_callback (Optional[Callable[[float], None]]): Optional callback function
            that receives generation progress as a float between 0 and 1
        images_b64 (Optional[List[str]]): Optional list of base64-encoded images for
            vision-language models
        stop_strings (Optional[List[str]]): Optional list of strings that will stop generation
            when encountered
        generate_args (dict): Dictionary of generation parameters that are passed to `mlx_lm`
            Full list of parameters:
            https://github.com/ml-explore/mlx-examples/blob/bd6d910/llms/mlx_lm/utils.py#L158
        top_logprobs (Optional[int]): Number of top token probabilities to return per token.
            Must be <= MAX_TOP_LOGPROBS

    Yields:
        GenerationResult: Named tuple containing:
            - text: Generated text segment
            - tokens: List of generated tokens, as TokenLogprob named tuples
            - top_logprobs: Token probability information if requested
            - stop_condition: Information about why generation stopped, if applicable

    Raises:
        ValueError: If top_logprobs exceeds MAX_TOP_LOGPROBS
    """
    set_seed(generate_args.pop("seed", None))

    # Process prompt
    generate_step_input = model_kit.process_prompt(
        prompt_tokens, images_b64, prompt_progress_callback, generate_args
    )

    # Add outlines logits processor if json_schema is provided
    logits_processor = []
    json_schema = generate_args.pop("json_schema", None)
    is_structured_output_request = json_schema is not None
    if is_structured_output_request:
        logits_processor.append(OutlinesLogitsProcessor(model_kit, json_schema))
    generate_args["logits_processor"] = logits_processor

    if top_logprobs is None:
        top_logprobs = 0
    if top_logprobs > MAX_TOP_LOGPROBS:
        raise ValueError(
            f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
        )

    max_tokens = generate_args.pop("max_tokens")
    tokenizer = model_kit.tokenizer
    detokenizer = model_kit.detokenizer
    detokenizer.reset()
    # keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[TokenLogprob] = []
    top_logprobs_buffer: List[List[TokenLogprob]] = []

    stop_sequences = [
        tokenize(model_kit, sequence) for sequence in (stop_strings or [])
    ]
    stop_processor = StopProcessor(tokenizer, stop_sequences)
    stop_processor_result = None

    for (token, logprobs), n in zip(
        mlx_lm.utils.generate_step(
            generate_step_input, model_kit.model, **generate_args
        ),
        range(max_tokens),
    ):
        model_kit.record_generated_token(token)
        detokenizer.add_token(token)
        token_buffer.append(TokenLogprob(token, tokenizer.decode(token), float(logprobs[token])))
        if top_logprobs:
            top_logprobs_buffer.append(
                summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
            )
        stop_processor_result = stop_processor.process_token(token)

        if stop_processor_result.status == "full_stop":
            break
        # If we currently have generated a partial match with a stop sequence, generate new
        # tokens until we know if the stop sequence is hit or not (i.e., make sure not to yield yet)
        if stop_processor_result.status == "partial_match":
            continue

        # only yield a generation result the detokenizer has a segment to yield
        new_text = detokenizer.last_segment
        if new_text:
            yield GenerationResult(
                text=new_text,
                tokens=token_buffer,
                stop_condition=None,
                top_logprobs=top_logprobs_buffer,
            )
            token_buffer = []
            top_logprobs_buffer = []

    # check is there any remaining text to send
    detokenizer.finalize()
    last_segment = detokenizer.last_segment
    last_segment, generation_stop_condition = stop_processor.finalize(
        last_segment, stop_processor_result
    )
    yield GenerationResult(
        text=last_segment,
        tokens=token_buffer,
        stop_condition=generation_stop_condition,
        top_logprobs=top_logprobs_buffer,
    )


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    """
    Convert a text prompt into a list of token IDs using the model's tokenizer.

    Args:
        model_kit (ModelKit | VisionModelKit): The model kit containing the tokenizer
        prompt (str): The text prompt to tokenize

    Returns:
        List[int]: List of token IDs representing the tokenized prompt
    """
    return model_kit.tokenize(prompt)
