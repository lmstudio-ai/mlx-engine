import argparse
import base64
import time
from typing import List

from mlx_engine.generate import load_model, load_draft_model, create_generator, tokenize
from mlx_engine.utils.token import Token
from mlx_engine.model_kit import VALID_KV_BITS, VALID_KV_GROUP_SIZE


DEFAULT_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Explain the rules of chess in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
DEFAULT_TEMP = 0.8


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="LM Studio mlx-engine inference script"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The file system path to the model",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        type=str,
        help="Message to be processed by the model",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Path of the images to process",
    )
    parser.add_argument(
        "--temp",
        default=DEFAULT_TEMP,
        type=float,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--stop-strings",
        type=str,
        nargs="+",
        help="Strings that will stop the generation",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=0,
        help="Number of top logprobs to return",
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
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        help="When --kv-bits is set, start quantizing the KV cache from this step onwards",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="The file system path to the draft model for speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
    )
    parser.add_argument(
        "--print-prompt-progress",
        action="store_true",
        help="Enable printed prompt processing progress callback",
    )
    return parser


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None

    def add_tokens(self, tokens: List[Token]):
        """Record new tokens and their timing."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
        for token in tokens:
            if token.from_draft:
                if self.num_accepted_draft_tokens is None:
                    self.num_accepted_draft_tokens = 0
                self.num_accepted_draft_tokens += 1
        self.total_tokens += len(tokens)

    def print_stats(self):
        """Print generation statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time
        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = self.total_tokens / effective_time if effective_time > 0 else float("inf")
        print(f"\n\nGeneration stats:")
        print(f" - Tokens per second: {tokens_per_second:.2f}")
        if self.num_accepted_draft_tokens is not None:
            print(f" - Number of accepted draft tokens: {self.num_accepted_draft_tokens}")
        print(f" - Time to first token: {time_to_first_token:.2f}s")
        print(f" - Total tokens generated: {self.total_tokens}")
        print(f" - Total time: {total_time:.2f}s")



if __name__ == "__main__":
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    if isinstance(args.images, str):
        args.images = [args.images]

    # Set up prompt processing callback
    def prompt_progress_callback(percent):
        if args.print_prompt_progress:
            width = 40  # bar width
            filled = int(width * percent / 100)
            bar = "█" * filled + "░" * (width - filled)
            print(f"\rProcessing prompt: |{bar}| ({percent:.1f}%)", end="", flush=True)
            if percent >= 100:
                print()  # new line when done
        else:
            pass

    # Load the model
    model_path = args.model
    model_kit = load_model(
        str(model_path),
        max_kv_size=args.max_kv_size,
        trust_remote_code=False,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )

    # Load draft model if requested
    if args.draft_model:
        load_draft_model(model_kit=model_kit, path=args.draft_model)

    # Tokenize the prompt
    prompt = args.prompt
    prompt_tokens = tokenize(model_kit, prompt)

    # Handle optional images
    images_base64 = []
    if args.images:
        if isinstance(args.images, str):
            args.images = [args.images]
        images_base64 = [image_to_base64(img_path) for img_path in args.images]

    # Record top logprobs
    logprobs_list = []

    # Initialize generation stats collector
    stats_collector = GenerationStatsCollector()

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_base64,
        stop_strings=args.stop_strings,
        max_tokens=1024,
        top_logprobs=args.top_logprobs,
        prompt_progress_callback=prompt_progress_callback,
        num_draft_tokens=args.num_draft_tokens,
        temp=args.temp,
    )
    for generation_result in generator:
        print(generation_result.text, end="", flush=True)
        stats_collector.add_tokens(generation_result.tokens)
        logprobs_list.extend(generation_result.top_logprobs)

        if generation_result.stop_condition:
            stats_collector.print_stats()
            print(
                f"\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
            )
            if generation_result.stop_condition.stop_string:
                print(f"Stop string: {generation_result.stop_condition.stop_string}")

    if args.top_logprobs:
        [print(x) for x in logprobs_list]
