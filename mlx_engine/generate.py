from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from typing import Iterator, List, Literal, NamedTuple, Optional
import json
import logging
from pathlib import Path
import sys

from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load as mlx_lm_load
from mlx_lm.models.cache import make_prompt_cache

from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)
from mlx_engine.utils.token import Token
from mlx_engine.utils.eot_tokens import sanitize_eos_tokens
from mlx_engine.utils.top_logprobs import summarize_top_logprobs
from mlx_engine.stop_string_processor import (
    StopStringProcessor,
    StopStringProcessorResult,
)
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.speculative_decoding import (
    determine_draft_model_for_generation,
    configure_num_draft_tokens_in_generate_args,
)
from outlines.processors.structured import JSONLogitsProcessor
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.cache_wrapper import PROMPT_PROCESSING_CHUNK_SIZE
from mlx_engine.utils.prompt_progress_reporter import (
    BatchedMlxLmReporterAdapter,
    LoggerReporter,
    PromptProgressReporter,
    DefaultPromptProgressReporter,
    MlxLmReporterAdapter,
    StopPromptProcessing,
)

MAX_TOP_LOGPROBS = 10

StopReason = Literal["eos_token", "stop_string", "user_cancelled"]

logger = logging.getLogger(__name__)


class GenerationStopCondition(NamedTuple):
    stop_reason: StopReason
    stop_string: str
    # sequence of token ids that the stop string was found in
    stop_tokens: List[int]


class GenerationResult(NamedTuple):
    text: str
    tokens: List[Token]
    top_logprobs: List[List[Token]]
    stop_condition: Optional[GenerationStopCondition]


def construct_user_cancelled_result():
    return GenerationResult(
        text="",
        tokens=[],
        top_logprobs=[],
        stop_condition=GenerationStopCondition(
            stop_reason="user_cancelled",
            stop_string="",
            stop_tokens=[],
        ),
    )


def _handle_stop_string_detected(
    tokenizer,
    stop_string_processor_result: StopStringProcessorResult,
    text: str,
    token_buffer: List[Token],
    top_logprobs_buffer: List[List[Token]],
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


def load_model(
    model_path: str | Path,
    *,
    vocab_only: bool = False,
    max_kv_size: int | None = 4096,
    max_seq_nums: int | None = 4,
    trust_remote_code: bool = False,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
) -> ModelKit | VisionModelKit:
    """
    Load a language model or vision-language model from the specified path.

    This function determines the model type based on the config.json file in the model directory
    and initializes either a standard language model or a vision-language model accordingly.

    Args:
        model_path (str | Path): Path to the model directory containing model files and config.json.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        max_seq_nums (int): The maximum number of parallel generation requests that can be worked on
        trust_remote_code (bool): Whether to allow loading of remote code during model initialization.
        kv_bits (Optional[int]): Number of bits for KV cache quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled.

    Returns:
        ModelKit | VisionModelKit: An initialized model instance:
            - ModelKit: for text-only models and vision models with vision add-on support
            - VisionModelKit: for vision models that are not yet supported by ModelKit

    Raises:
        FileNotFoundError: If config.json is not found in the specified model path
        json.JSONDecodeError: If config.json exists but contains invalid JSON
        ValueError: If the model configuration is invalid or unsupported
    """
    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())
    model_type = config_json.get("model_type", None)

    # only use VisionModelKit if ModelKit doesn't have vision support for this model
    if "vision_config" in config_json and not ModelKit.is_supported_vision_arch(
        model_type
    ):
        if any([kv_bits, kv_group_size, quantized_kv_start]):
            raise ValueError(
                "MLX vision models do not currently support KV cache quantization"
            )
        model_kit = VisionModelKit(model_path, vocab_only, trust_remote_code)
    else:
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )
        is_batchable = True
        model, _ = mlx_lm_load(model_path, lazy=True)
        is_batchable &= all(hasattr(c, "merge") for c in make_prompt_cache(model))
        del model
        is_batchable &= kv_bits is None
        is_batchable &= "vision_config" not in config_json
        if is_batchable:
            model_kit = BatchedModelKit(
                model_path,
                max_kv_size=max_kv_size,
                max_seq_nums=max_seq_nums,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            )
        else:
            model_kit = ModelKit(
                model_path,
                vocab_only,
                max_kv_size=max_kv_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            )
    sanitize_eos_tokens(model_kit)
    model_kit.start()
    return model_kit


def load_draft_model(model_kit: ModelKit | VisionModelKit, path: str | Path) -> None:
    model_kit.load_draft_model(path)


def is_draft_model_compatible(
    model_kit: ModelKit | VisionModelKit, path: str | Path
) -> bool:
    return model_kit.is_draft_model_compatible(path)


def unload_draft_model(model_kit: ModelKit | VisionModelKit) -> None:
    model_kit.unload_draft_model()


def create_generator(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit,
    prompt_tokens: List[int],
    **kwargs,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    This function sets up and manages the text generation process, handling various generation
    parameters, processing callbacks, and managing generation constraints. It supports both
    standard language models and vision-language models.

    Args:
        model_kit (ModelKit | VisionModelKit): The initialized model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_reporter (Optional[PromptProgressReporter]): Reporter for receiving prompt
            processing progress updates. Reporter methods should return True to continue processing,
            or False to stop generation
        images_b64 (Optional[List[str]]): List of base64-encoded images for vision-language models
        max_image_size (Optional[tuple[int, int]]): Maximum dimensions (width, height) for images.
            Images will be resized to fit within these dimensions while maintaining aspect ratio if
            they exceed this size. If None, no resizing.
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
        top_k (Optional[int]): Top-k sampling parameter
        min_p (Optional[float]): Minimum probability threshold for token sampling
        min_tokens_to_keep (Optional[int]): Minimum number of tokens to keep during sampling
        seed (Optional[int]): Random seed for reproducible generation
        json_schema (Optional[str]): JSON schema for structured output generation
        max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to 10000000
        speculative_decoding_toggle (Optional[bool]): If not set, use speculative decoding
            if a draft model is loaded. If set to true, draft model must be loaded or else error.
            If set to false, speculative decoding is disabled even if a draft model is loaded.
        num_draft_tokens (Optional[int]): Number of tokens to draft when using speculative decoding

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
    if isinstance(model_kit, BatchedModelKit):
        return _batched_generation(model_kit, prompt_tokens, **kwargs)
    return _sequential_generation(model_kit, prompt_tokens, **kwargs)


def _sequential_generation(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
):
    with model_kit.generation_lock:
        set_seed(seed)

        generate_args = {}
        if prompt_progress_reporter is None:
            prompt_progress_reporter = LoggerReporter()

        # Set up kv cache
        if type(model_kit) is not VisionModelKit:
            for attr in [
                "max_kv_size",
                "kv_bits",
                "kv_group_size",
                "quantized_kv_start",
            ]:
                value = getattr(model_kit, attr, None)
                if value is not None:
                    generate_args[attr] = value

        # Set up repetition penalty
        repetition_penalty_kwargs = {}
        if repetition_penalty is not None:
            repetition_penalty_kwargs["repetition_penalty"] = repetition_penalty
            if repetition_context_size is not None:
                repetition_penalty_kwargs["repetition_context_size"] = (
                    repetition_context_size
                )

        # Set up speculative decoding
        draft_model = determine_draft_model_for_generation(
            model_kit, speculative_decoding_toggle
        )
        configure_num_draft_tokens_in_generate_args(
            model_kit, draft_model, num_draft_tokens, generate_args
        )

        # Process prompt
        try:
            input_tokens, input_embeddings = model_kit.process_prompt(
                prompt_tokens,
                images_b64,
                prompt_progress_reporter,
                generate_args,
                max_image_size,
                speculative_decoding_toggle,
            )
        except StopPromptProcessing:
            yield construct_user_cancelled_result()
            return
        if draft_model is None:
            # input embeddings not yet supported for speculative decoding in mlx-lm
            generate_args["input_embeddings"] = input_embeddings

        # Setup logits processors
        logits_processors = []
        if repetition_penalty and repetition_penalty != 0.0:
            cached_tokens = (
                prompt_tokens[: -len(input_tokens)]
                if len(input_tokens) > 0
                else prompt_tokens
            )
            logits_processors.append(
                RepetitionPenaltyProcessor(
                    token_history=cached_tokens, **repetition_penalty_kwargs
                )
            )

        # Set up sampler
        generate_args["sampler"] = make_sampler(
            **{
                k: v
                for k, v in {
                    "temp": temp,
                    "top_p": top_p,
                    "min_p": min_p,
                    "min_tokens_to_keep": min_tokens_to_keep,
                    "top_k": top_k,
                }.items()
                if v is not None
            }
        )

        # If using VisionModelKit, immediately record the token once it's sampled
        if type(model_kit) is VisionModelKit:
            sampler_func = generate_args["sampler"]

            def sampler_func_wrapper(*args, **kwargs):
                token = sampler_func(*args, **kwargs)
                model_kit.record_sampled_token(token)
                return token

            generate_args["sampler"] = sampler_func_wrapper

        # Validate top_logprobs
        if top_logprobs is None:
            top_logprobs = 0
        if top_logprobs > MAX_TOP_LOGPROBS:
            raise ValueError(
                f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
            )

        # Keep track of tokens buffered by detokenizer to yield accurate generation results
        token_buffer: List[Token] = []
        top_logprobs_buffer: List[List[Token]] = []

        tokenizer = model_kit.tokenizer

        # Add outlines logits processor if json_schema is provided
        is_structured_output_request = json_schema is not None
        if is_structured_output_request:
            logits_processors.append(
                JSONLogitsProcessor(
                    json_schema,
                    OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                    tensor_library_name="mlx",
                )
            )

        # Set up stop string processor if non-empty stop_strings are provided
        stop_string_processor = None
        if stop_strings is not None and len(stop_strings) > 0:
            stop_string_processor = StopStringProcessor(stop_strings, tokenizer)
        text = ""

        # Determine callback for mlx-lm based on processing mode
        # When cache is NOT active (vision prompts), stream_generate handles prompt processing
        # When cache IS active (text-only), cache_wrapper already handled it
        if not model_kit.is_cross_prompt_cache_active():
            mlx_lm_callback = MlxLmReporterAdapter(
                prompt_progress_reporter, emit_begin=True
            )
        else:
            mlx_lm_callback = None

        stream = stream_generate(
            model=model_kit.model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=input_tokens,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            prompt_progress_callback=mlx_lm_callback,
            prefill_step_size=PROMPT_PROCESSING_CHUNK_SIZE,
            **generate_args,
        )

        while True:
            try:
                generation_result = next(stream)
            except StopIteration:
                break
            except StopPromptProcessing:
                yield construct_user_cancelled_result()
                return

            # Token processor
            token = generation_result.token
            text += generation_result.text
            # record generated token to cache, if cache is active
            if model_kit.is_cross_prompt_cache_active():
                model_kit.record_token_to_cache(token)

            logprobs = generation_result.logprobs
            token_buffer.append(
                Token(
                    token,
                    tokenizer.decode(token),
                    float(logprobs[token]),
                    from_draft=generation_result.from_draft,
                )
            )
            if top_logprobs:
                top_logprobs_buffer.append(
                    summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
                )

            # Stop processor
            if stop_string_processor is not None:
                stop_string_processor_result = stop_string_processor.process_token(
                    token
                )
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

            # Standard yield - yield when a non-empty text segment is available or eos token is hit
            if text or token in tokenizer.eos_token_ids:
                # populate stop_condition if we hit an eos token
                stop_condition = None
                if token in tokenizer.eos_token_ids:
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


def _batched_generation(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
    request_id: str | None = None,
) -> Iterator[GenerationResult]:
    input_tokens = prompt_tokens

    if prompt_progress_reporter is None:
        prompt_progress_reporter = DefaultPromptProgressReporter()

    # Set up repetition penalty
    repetition_penalty_kwargs = {}
    if repetition_penalty is not None:
        repetition_penalty_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            repetition_penalty_kwargs["repetition_context_size"] = (
                repetition_context_size
            )

    # Setup logits processors
    logits_processors = []
    if repetition_penalty and repetition_penalty != 0.0:
        cached_tokens = (
            prompt_tokens[: -len(input_tokens)]
            if len(input_tokens) > 0
            else prompt_tokens
        )
        logits_processors.append(
            RepetitionPenaltyProcessor(
                token_history=cached_tokens, **repetition_penalty_kwargs
            )
        )

    # Set up sampler
    sampler = make_sampler(
        **{
            k: v
            for k, v in {
                "temp": temp,
                "top_p": top_p,
                "min_p": min_p,
                "min_tokens_to_keep": min_tokens_to_keep,
                "top_k": top_k,
            }.items()
            if v is not None
        }
    )

    # Validate top_logprobs
    if top_logprobs is None:
        top_logprobs = 0
    if top_logprobs > MAX_TOP_LOGPROBS:
        raise ValueError(
            f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
        )

    # Keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[Token] = []
    top_logprobs_buffer: List[List[Token]] = []

    tokenizer = model_kit.tokenizer

    # Add outlines logits processor if json_schema is provided
    is_structured_output_request = json_schema is not None
    if is_structured_output_request:
        logits_processors.append(
            JSONLogitsProcessor(
                json_schema,
                OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                tensor_library_name="mlx",
            )
        )

    # Set up stop string processor if non-empty stop_strings are provided
    stop_string_processor = None
    if stop_strings is not None and len(stop_strings) > 0:
        stop_string_processor = StopStringProcessor(stop_strings, tokenizer)
    text = ""

    mlx_lm_callback = BatchedMlxLmReporterAdapter(
        prompt_progress_reporter, emit_begin=True
    )

    stream = model_kit.generate(
        prompt_tokens=input_tokens,
        request_id=request_id,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_progress_callback=mlx_lm_callback,
        top_logprobs=top_logprobs,
    )

    while True:
        try:
            generation_result = next(stream)
        except StopIteration:
            break
        # TODO: implement this
        # except StopPromptProcessing:
        #     yield construct_user_cancelled_result()
        #     return

        # Token processor
        token = generation_result.token
        text += generation_result.text

        token_buffer.append(
            Token(
                token,
                tokenizer.decode(token),
                generation_result.token_logprob,
                from_draft=generation_result.from_draft,
            )
        )
        if top_logprobs and generation_result.top_logprobs is not None:
            top_logprobs_buffer.append(generation_result.top_logprobs)

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

        # Standard yield - yield when a non-empty text segment is available or eos token is hit
        if text or token in tokenizer.eos_token_ids:
            # populate stop_condition if we hit an eos token
            stop_condition = None
            if token in tokenizer.eos_token_ids:
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


def stop_generation(model_kit: BatchedModelKit, request_id: str):
    """
    Register stop request based off of request_id. For now, this is only supported for `BatchedModelKit`, but this may be extended in the future
    """
    if not isinstance(model_kit, BatchedModelKit):
        logger.error(
            f"cannot cancel {request_id=}, this API is only available during batched generation"
        )
        return
    if request_id is None or request_id == "":
        logger.error("request_id cannot be empty in stop request")
    model_kit.remove(request_id)


def unload(model_kit: ModelKit | VisionModelKit | BatchedModelKit):
    if isinstance(model_kit, BatchedModelKit):
        model_kit.shutdown()
