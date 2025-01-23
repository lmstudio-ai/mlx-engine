import argparse
import base64
import time

from mlx_engine.generate import load_model, load_draft_model, create_generator, tokenize
from mlx_engine.model_kit import VALID_KV_BITS, VALID_KV_GROUP_SIZE


DEFAULT_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Explain the rules of chess in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="LM Studio mlx-engine inference script"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The path to the local model directory.",
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
        help="The path to the local model directory to use as the draft model for speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
    )
    parser.add_argument(
        "--print-prompt-processing",
        action="store_true",
        help="Enable printed prompt processing callback",
    )
    return parser


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    if isinstance(args.images, str):
        args.images = [args.images]

    # Set up prompt processing callback
    def prompt_progress_callback(percent):
        if args.print_prompt_processing:
            print(f"Processed {percent}% of prompt")
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
        load_draft_model(model_kit=model_kit, draft_model_path=args.draft_model)

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

    # Initialize timing variables
    start_time = time.time()
    first_token_time = None
    total_tokens = 0

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
    )
    for generation_result in generator:
        if first_token_time is None:
            first_token_time = time.time()

        print(generation_result.text, end="", flush=True)
        total_tokens += 1
        logprobs_list.extend(generation_result.top_logprobs)

        if generation_result.stop_condition:
            end_time = time.time()
            total_time = end_time - start_time
            tokens_per_second = total_tokens / total_time

            print(
                f"\n\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
            )
            if generation_result.stop_condition.stop_string:
                print(f"Stop string: {generation_result.stop_condition.stop_string}")

            ttft = first_token_time - start_time
            print(f"\nGeneration stats:")
            print(f" - Time to first token: {ttft:.2f}s")
            print(f" - Total tokens generated: {total_tokens}")
            print(f" - Total time: {total_time:.2f}s")
            print(f" - Tokens per second: {tokens_per_second:.2f}")

    if args.top_logprobs:
        [print(x) for x in logprobs_list]
