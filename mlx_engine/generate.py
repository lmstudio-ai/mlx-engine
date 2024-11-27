from typing import Callable, Iterator, List, Literal, NamedTuple, Optional
import json
from pathlib import Path
import sys

import mlx_lm

from mlx_engine.model_kit import ModelKit
from mlx_engine.vision.vision_model_kit import VisionModelKit
from mlx_engine.processors.outlines_logits_processor import OutlinesLogitsProcessor
from mlx_engine.utils.top_logprobs import summarize_top_logprobs, TokenLogprob
from mlx_engine.stop_string_processor import (
    StopStringProcessor,
    StopStringProcessorResult,
)
from mlx_engine.utils.set_seed import set_seed


MAX_TOP_LOGPROBS = 10

StopReason = Literal["eos_token", "stop_string"]


class GenerationStopCondition(NamedTuple):
    stop_reason: StopReason
    stop_string: str
    # sequence of token ids that the stop string was found in
    stop_tokens: List[int]


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

    This function determines the model type based on the config.json file in the model directory
    and initializes either a standard language model or a vision-language model accordingly.

    Args:
        model_path (str | Path): Path to the model directory containing model files and config.json.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        trust_remote_code (bool): Whether to allow loading of remote code during model initialization.

    Returns:
        ModelKit | VisionModelKit: An initialized model instance:
            - ModelKit for standard language models
            - VisionModelKit for models that can process both text and images

    Raises:
        FileNotFoundError: If config.json is not found in the specified model path
        json.JSONDecodeError: If config.json exists but contains invalid JSON
        ValueError: If the model configuration is invalid or unsupported
    """
    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())

    if "vision_config" in config_json:
        return VisionModelKit(model_path, max_kv_size, trust_remote_code)
    else:
        return ModelKit(model_path, max_kv_size)


def create_generator(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_callback: Optional[Callable[[float], None]] = None,
    images_b64: Optional[List[str]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    This function sets up and manages the text generation process, handling various generation
    parameters, processing callbacks, and managing generation constraints. It supports both
    standard language models and vision-language models.

    Args:
        model_kit (ModelKit | VisionModelKit): The initialized model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_callback (Optional[Callable[[float], None]]): Callback function that receives
            generation progress as a float between 0 and 1
        images_b64 (Optional[List[str]]): List of base64-encoded images for vision-language models
        stop_strings (Optional[List[str]]): List of strings that will trigger generation to stop
            when encountered
        top_logprobs (Optional[int]): Number of top token probabilities to return per token
            Must be <= MAX_TOP_LOGPROBS
        repetition_penalty (Optional[float]): Penalty factor for repeated tokens. Higher values
            discourage repetition
        repetition_context_size (Optional[int]): Number of previous tokens to consider for
            repetition penalty. Defaults to 20
        temp (Optional[float]): Temperature for sampling. Higher values increase randomness
        top_p (Optional[float]): Top-p (nucleus) sampling parameter
        min_p (Optional[float]): Minimum probability threshold for token sampling
        min_tokens_to_keep (Optional[int]): Minimum number of tokens to keep during sampling
        seed (Optional[int]): Random seed for reproducible generation
        json_schema (Optional[str]): JSON schema for structured output generation
        max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to 10000000

    Yields:
        GenerationResult: A named tuple containing:
            - text (str): Generated text segment
            - tokens (List[TokenLogprob]): List of generated tokens with their probabilities
            - top_logprobs (List[List[TokenLogprob]]): Token probability information if requested
            - stop_condition (Optional[GenerationStopCondition]): Information about why
              generation stopped, if applicable

    Raises:
        ValueError: If top_logprobs exceeds MAX_TOP_LOGPROBS or if any parameters are invalid
    """
    set_seed(seed)

    generate_args = {}
    if type(model_kit) is not VisionModelKit:
        generate_args["max_kv_size"] = model_kit.max_kv_size

    # Set up repetition penalty
    repetition_penalty_kwargs = {}
    if repetition_penalty is not None:
        repetition_penalty_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            repetition_penalty_kwargs["repetition_context_size"] = (
                repetition_context_size
            )
    generate_args["logits_processors"] = mlx_lm.utils.make_logits_processors(
        logit_bias=None,
        **repetition_penalty_kwargs,
    )

    # Process prompt
    stream_generate_input = model_kit.process_prompt(
        prompt_tokens,
        images_b64,
        prompt_progress_callback,
        repetition_context_size,
        generate_args,
    )

    # Workaround until mlx_lm.utils.stream_generate supports prompt as type mx.array
    stream_generate_input = stream_generate_input.tolist()

    # Set up sampler
    generate_args["sampler"] = mlx_lm.utils.make_sampler(
        **{
            k: v
            for k, v in {
                "temp": temp,
                "top_p": top_p,
                "min_p": min_p,
                "min_tokens_to_keep": min_tokens_to_keep,
            }.items()
            if v is not None
        }
    )

    # For vision models, immediately record the token once it's sampled
    if type(model_kit) is VisionModelKit:
        sampler_func = generate_args["sampler"]

        def sampler_func_wrapper(*args, **kwargs):
            token = sampler_func(*args, **kwargs)
            model_kit.record_sampled_token(token)
            return token

        generate_args["sampler"] = sampler_func_wrapper

    # Add outlines logits processor if json_schema is provided
    is_structured_output_request = json_schema is not None
    if is_structured_output_request:
        generate_args["logits_processors"].append(
            OutlinesLogitsProcessor(model_kit, json_schema)
        )

    # Validate top_logprobs
    if top_logprobs is None:
        top_logprobs = 0
    if top_logprobs > MAX_TOP_LOGPROBS:
        raise ValueError(
            f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
        )

    # Keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[TokenLogprob] = []
    top_logprobs_buffer: List[List[TokenLogprob]] = []

    tokenizer = model_kit.tokenizer

    # Set up stop string processor if non-empty stop_strings are provided
    stop_string_processor = None
    if stop_strings is not None and len(stop_strings) > 0:
        stop_string_processor = StopStringProcessor(stop_strings, tokenizer)
    text = ""

    def _handle_stop_string_detected(
        tokenizer,
        stop_string_processor_result: StopStringProcessorResult,
        text: str,
        token_buffer: List[TokenLogprob],
        top_logprobs_buffer: List[List[TokenLogprob]],
    ) -> GenerationResult:
        """
        Helper method to Handle completion of text generation when a stop string is
        encountered.

        Args:
            tokenizer: The tokenizer instance
            stop_string_processor_result: Result from stop string processor
            text: Current generated text
            token_buffer: Buffer of generated tokens
            top_logprobs_buffer: Buffer of token probabilities

        Returns:
            GenerationResult: Final generation result including stop condition
        """
        # Finalize detokenizer to get remaining text
        detokenizer = tokenizer.detokenizer
        detokenizer.finalize()
        text += detokenizer.last_segment

        # Process stop string by trimming text segment where it begins
        stop_string = stop_string_processor_result.stop_string
        stop_string_start_pos = text.find(stop_string)

        if stop_string_start_pos != -1:
            text = text[:stop_string_start_pos]
        else:
            # this is known to happen when the eos token is a stop string
            sys.stderr.write(
                f"[mlx-engine] Stop string '{stop_string}' not found in final text segment, "
                "even though a full stop was detected. Not trimming final segment."
            )

        stop_condition = GenerationStopCondition(
            stop_reason="stop_string",
            stop_string=stop_string,
            stop_tokens=stop_string_processor_result.stop_tokens,
        )

        return GenerationResult(
            text=text,
            tokens=token_buffer,
            stop_condition=stop_condition,
            top_logprobs=top_logprobs_buffer,
        )

    for generation_result in mlx_lm.utils.stream_generate(
        model=model_kit.model,
        tokenizer=tokenizer,
        prompt=stream_generate_input,
        max_tokens=max_tokens,
        **generate_args,
    ):
        # Token processor
        token = generation_result.token
        text += generation_result.text
        model_kit.update_cache_wrapper(token)

        logprobs = generation_result.logprobs
        token_buffer.append(
            TokenLogprob(token, tokenizer.decode(token), float(logprobs[token]))
        )
        if top_logprobs:
            top_logprobs_buffer.append(
                summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
            )

        # Stop processor
        if stop_string_processor is not None:
            stop_string_processor_result = stop_string_processor.process_token(token)
            if stop_string_processor_result.status == "full_stop":
                yield _handle_stop_string_detected(
                    tokenizer,
                    stop_string_processor_result,
                    text,
                    token_buffer,
                    top_logprobs_buffer,
                )
                break  # stop generation

            # If we currently have generated a partial match with a stop sequence, or detected an
            # in-progress multi-byte string, generate new tokens until we know if the stop sequence
            # is hit or not (i.e., make sure not to yield yet)
            if (
                stop_string_processor_result.status == "partial_match"
                or stop_string_processor_result.status == "multi_byte"
            ):
                continue

        # Standard yield - only yield when a non-empty text segment is available
        if text:
            # populate stop_condition if we hit an eos token
            stop_condition = None
            if token == tokenizer.eos_token_id:
                stop_condition = GenerationStopCondition(
                    stop_reason="eos_token",
                    stop_string=tokenizer.decode(token),
                    stop_tokens=[token],
                )
            yield GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=stop_condition,
                top_logprobs=top_logprobs_buffer,
            )
            token_buffer = []
            top_logprobs_buffer = []
            text = ""


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    """
    Convert a text prompt into a list of token IDs using the model's tokenizer.

    Args:
        model_kit (ModelKit | VisionModelKit): The model kit instance containing the tokenizer
            to use for tokenization
        prompt (str): The raw text prompt to be tokenized

    Returns:
        List[int]: A list of integer token IDs representing the tokenized prompt,
            ready for model input
    """
    return model_kit.tokenize(prompt)
