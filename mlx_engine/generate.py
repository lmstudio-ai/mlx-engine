import json
import logging
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Literal, NamedTuple, Optional, Union

from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from outlines.processors.structured import JSONLogitsProcessor

from mlx_engine.cache_wrapper import PROMPT_PROCESSING_CHUNK_SIZE, StopPromptProcessing
from mlx_engine.utils.progress_decorators import backward_compatible
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)
from mlx_engine.stop_string_processor import (
    StopStringProcessor,
    StopStringProcessorResult,
)
from mlx_engine.utils.eot_tokens import get_eot_token_ids
from mlx_engine.utils.hardware import PerformanceProfileCompat
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.utils.progress_decorators import (
    backward_compatible,
    ratchet,
    throw_to_stop,
    token_count,
)
from mlx_engine.utils.prompt_processing import plan_prefill_strategy
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.speculative_decoding import (
    configure_num_draft_tokens_in_generate_args,
    determine_draft_model_for_generation,
)
from mlx_engine.utils.token import Token
from mlx_engine.utils.top_logprobs import summarize_top_logprobs
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit

# Import metrics collection
try:
    from mlx_engine.utils.logger import get_global_structured_logger
    from mlx_engine.utils.metrics import get_global_collector

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

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


def load_model(
    model_path: str | Path,
    *,
    vocab_only: bool = False,
    max_kv_size: Optional[int] = 4096,
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
        return VisionModelKit(model_path, vocab_only, trust_remote_code)
    else:
        return ModelKit(
            model_path,
            vocab_only,
            max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )


def load_draft_model(model_kit: ModelKit | VisionModelKit, path: str | Path) -> None:
    model_kit.load_draft_model(path)


def is_draft_model_compatible(
    model_kit: ModelKit | VisionModelKit, path: str | Path
) -> bool:
    return model_kit.is_draft_model_compatible(path)


def unload_draft_model(model_kit: ModelKit | VisionModelKit) -> None:
    model_kit.unload_draft_model()


def create_generator(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_callback: Optional[Callable[[float], Union[bool, None]]] = None,
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
    prefill_mode: Optional[str] = None,
    performance_profile: Optional[PerformanceProfileCompat] = None,
    available_mem_gb: Optional[float] = None,
    max_prefill_tokens_per_pass: Optional[int] = None,
    enable_branching: Optional[bool] = None,
    cache_slots: Optional[int] = None,
    checkpoint_branch: Optional[str] = None,
    restore_branch: Optional[str] = None,
    release_branch: Optional[str] = None,
    pin_branch: Optional[str] = None,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    This function sets up and manages the text generation process, handling various generation
    parameters, processing callbacks, and managing generation constraints. It supports both
    standard language models and vision-language models.

    Args:
        model_kit (ModelKit | VisionModelKit): The initialized model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_callback (Optional[Callable[[float], Union[bool, None]]]): Callback function that receives
            generation progress as a float between 0 and 100. For backward compatibility, accepts both
            new-style callbacks that return True/False and old-style callbacks that return None or
            have no explicit return. All callbacks are treated as continuing processing unless they
            explicitly return False
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
        enable_branching (Optional[bool]): Enable branching cache support for O(1) branch switching
        cache_slots (Optional[int]): Number of cache slots for branching (default: 4)
        checkpoint_branch (Optional[str]): Branch ID to checkpoint after generation
        restore_branch (Optional[str]): Branch ID to restore before generation
        release_branch (Optional[str]): Branch ID to release before generation
        pin_branch (Optional[str]): Branch ID to pin before generation

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

    # Handle branching cache operations
    if enable_branching:
        if not hasattr(model_kit, "enable_branching_cache"):
            raise ValueError("Branching cache is not supported for this model type")

        # Initialize branching cache if not already enabled
        if not getattr(model_kit, "enable_branching", False):
            cache_slots = cache_slots or 4
            model_kit.enable_branching_cache(max_slots=cache_slots)

        # Handle branch operations before generation
        if restore_branch:
            try:
                model_kit.restore_branch(restore_branch)
                logger.info(f"Restored branch: {restore_branch}")
            except KeyError as e:
                logger.warning(f"Failed to restore branch {restore_branch}: {e}")

        if release_branch:
            try:
                model_kit.release_branch(release_branch)
                logger.info(f"Released branch: {release_branch}")
            except RuntimeError as e:
                logger.warning(f"Failed to release branch {release_branch}: {e}")

        if pin_branch:
            try:
                model_kit.pin_branch(pin_branch)
                logger.info(f"Pinned branch: {pin_branch}")
            except KeyError as e:
                logger.warning(f"Failed to pin branch {pin_branch}: {e}")

    generate_args = {}
    # Apply backward compatibility wrapper first to handle both old (None return) and new (bool return) callback patterns
    prompt_progress_callback = backward_compatible(prompt_progress_callback)

    # For each call to create_generator, wrap all prompt progress calls with a ratchet that
    # ensures reported progress monotonically increases. This is needed because prompt processing
    # occurs in different places depending on the model type and prompt content. The prompt will only
    # be processed once, but some contexts are not aware that the prompt is already processed, which
    # can cause the progress to look like it is being reset when it is actually already complete.
    # See https://github.com/lmstudio-ai/mlx-engine/issues/226.
    prompt_progress_callback = ratchet(prompt_progress_callback)

    # Set up kv cache
    if type(model_kit) is not VisionModelKit:
        for attr in ["max_kv_size", "kv_bits", "kv_group_size", "quantized_kv_start"]:
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

    # Plan prefill strategy if parameters provided
    prefill_plan = None
    if performance_profile is not None and available_mem_gb is not None:
        available_mem_bytes = int(available_mem_gb * 1024**3)

        # Convert string performance_profile to PerformanceProfileCompat if needed
        if isinstance(performance_profile, str):
            from mlx_engine.utils.hardware import (
                HardwareInfoCompat,
                select_profile_for_hardware,
            )

            # Create a fake hardware info for conversion
            fake_hardware = HardwareInfoCompat(
                model_identifier="Mac14,12",  # M3 Ultra identifier
                total_memory_gb=int(available_mem_gb),
                bandwidth_gbps=600,
                is_apple_silicon=True,
            )
            performance_profile = select_profile_for_hardware(
                fake_hardware, performance_profile, int(available_mem_gb)
            )

        kv_bytes_per_token = performance_profile.kv_bytes_per_token_estimate

        prefill_plan = plan_prefill_strategy(
            prompt_tokens=len(prompt_tokens),
            profile=performance_profile,
            kv_bytes_per_token=kv_bytes_per_token,
            available_mem_bytes=available_mem_bytes,
            requested_mode=prefill_mode,
            speculative_required=draft_model is not None,
        )

    # Initialize metrics collection if available
    metrics_collector = None
    structured_logger = None
    if METRICS_AVAILABLE:
        metrics_collector = get_global_collector()
        structured_logger = get_global_structured_logger()

        # Start prefill timing
        metrics_collector.start_prefill_timing()

    # Process prompt
    try:
        input_tokens, input_embeddings = model_kit.process_prompt(
            prompt_tokens,
            images_b64,
            prompt_progress_callback,
            generate_args,
            max_image_size,
            speculative_decoding_toggle,
        )

        # End prefill timing and record metrics
        if metrics_collector:
            metrics_collector.end_prefill_timing(len(prompt_tokens))

            # Log prefill performance
            if structured_logger:
                structured_logger.log_performance(
                    operation="prefill",
                    duration_s=metrics_collector.metrics.prefill_time_s,
                    tokens=len(prompt_tokens),
                    additional_data={
                        "mode": prefill_plan.mode if prefill_plan else "unknown",
                        "chunk_size": prefill_plan.chunk_size if prefill_plan else None,
                        "total_chunks": prefill_plan.total_chunks
                        if prefill_plan
                        else 1,
                    },
                )

    except StopPromptProcessing:
        if metrics_collector:
            metrics_collector.end_prefill_timing(len(prompt_tokens))
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

    # Add eot token ids to tokenizer
    tokenizer.eos_token_ids = tokenizer.eos_token_ids.union(
        get_eot_token_ids(tokenizer, model_kit.model_type)
    )

    # Set up stop string processor if non-empty stop_strings are provided
    stop_string_processor = None
    if stop_strings is not None and len(stop_strings) > 0:
        stop_string_processor = StopStringProcessor(stop_strings, tokenizer)
    text = ""

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

    # Use prefill plan chunk size if available, otherwise default
    prefill_step_size = (
        prefill_plan.chunk_size if prefill_plan else PROMPT_PROCESSING_CHUNK_SIZE
    )

    # Start generation timing
    if metrics_collector:
        metrics_collector.start_generation_timing()

    stream = stream_generate(
        model=model_kit.model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=input_tokens,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        prompt_progress_callback=token_count(throw_to_stop(prompt_progress_callback)),
        prefill_step_size=prefill_step_size,
        **generate_args,
    )

    generated_token_count = 0
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
        generated_token_count += 1

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

    # Finalize metrics collection
    if metrics_collector:
        metrics_collector.end_generation_timing(generated_token_count)
        metrics_collector.finalize_timing()

        # Update decision metrics
        if prefill_plan:
            metrics_collector.update_decision_metrics(
                mode=prefill_plan.mode,
                chunk_size=prefill_plan.chunk_size,
                total_chunks=prefill_plan.total_chunks,
                reason=prefill_plan.reason,
            )

        # Update cache stats if available
        if hasattr(model_kit, "cache_wrapper") and hasattr(
            model_kit.cache_wrapper, "get_cache_stats"
        ):
            cache_stats = model_kit.cache_wrapper.get_cache_stats()
            metrics_collector.update_cache_metrics(
                hits=cache_stats.hits,
                misses=cache_stats.misses,
                evictions=cache_stats.evictions,
                size_gb=cache_stats.size_gb,
                utilization_ratio=cache_stats.utilization_ratio,
            )

        # Log final metrics
        if structured_logger:
            structured_logger.log_metrics(metrics_collector.get_metrics().to_dict())

            # Log generation performance
            structured_logger.log_performance(
                operation="generation",
                duration_s=metrics_collector.metrics.generation_time_s,
                tokens=generated_token_count,
                additional_data={
                    "total_tokens": metrics_collector.metrics.total_tokens,
                    "tokens_per_second": metrics_collector.metrics.tokens_per_second,
                    "prefill_tokens_per_second": metrics_collector.metrics.prefill_tokens_per_second,
                },
            )

    # Handle post-generation branch checkpoint
    if enable_branching and checkpoint_branch:
        try:
            model_kit.checkpoint_branch(checkpoint_branch)
            logger.info(f"Checkpointed branch: {checkpoint_branch}")
        except RuntimeError as e:
            logger.warning(f"Failed to checkpoint branch {checkpoint_branch}: {e}")


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


def cli_parser():
    """
    Create and return the CLI argument parser for mlx-engine.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all flags
    """
    import argparse

    from mlx_engine.utils.kv_cache_quantization import (
        VALID_KV_BITS,
        VALID_KV_GROUP_SIZE,
    )

    parser = argparse.ArgumentParser(
        description="LM Studio mlx-engine inference script",
        epilog="""
Examples:
  # Basic usage with auto performance profile
  python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

  # High-performance mode for M3 Ultra with 512GB RAM
  python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --profile m3_ultra_512 --prefill-mode unbounded

  # Adaptive chunking with custom progress interval
  python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --profile m3_max_128 --adaptive-chunk --progress-interval-ms 500

  # Branching cache with performance optimization
  python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
    --kv-branching --cache-slots 8 --profile m3_ultra_256
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic arguments
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The file system path to the model",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the rules of chess in one sentence",
        type=str,
        help="Message to be processed by the model",
    )
    parser.add_argument(
        "--temp",
        default=0.8,
        type=float,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Max context size of the model",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=VALID_KV_BITS,
        help="Number of bits for KV cache quantization. Must be between 3 and 8 (inclusive)",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        choices=VALID_KV_GROUP_SIZE,
        help="Group size for KV cache quantization",
    )

    # Performance optimization arguments
    performance_group = parser.add_argument_group(
        "Performance Optimization",
        "Flags for optimizing performance on high-bandwidth Apple Silicon",
    )
    performance_group.add_argument(
        "--profile",
        choices=["auto", "default_safe", "m3_ultra_512", "m3_max_128", "m3_pro_64"],
        default="auto",
        help="Performance profile for hardware optimization (default: auto)",
    )
    performance_group.add_argument(
        "--prefill-mode",
        choices=["auto", "unbounded", "chunked"],
        default="auto",
        help="Prefill strategy mode (default: auto)",
    )
    performance_group.add_argument(
        "--adaptive-chunk",
        action="store_true",
        help="Enable adaptive chunk sizing for memory-efficient prefill",
    )
    performance_group.add_argument(
        "--kv-branching",
        action="store_true",
        help="Alias for --enable-branching (O(1) branch switching)",
    )
    performance_group.add_argument(
        "--progress-interval-ms",
        type=int,
        default=1000,
        help="Progress update interval in milliseconds (default: 1000)",
    )
    performance_group.add_argument(
        "--max-prefill-tokens",
        type=int,
        help="Maximum tokens per prefill pass (profile-dependent if not specified)",
    )

    return parser
