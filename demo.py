import argparse
import base64

from mlx_engine.generate import load_model, create_generator, tokenize


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

    # Load the model
    model_path = args.model
    model_kit = load_model(str(model_path), max_kv_size=4096, trust_remote_code=False)

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

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        None,
        images_base64,
        args.stop_strings,
        {"max_tokens": 1024},
        top_logprobs=args.top_logprobs,
    )
    for generation_result in generator:
        print(generation_result.text, end="", flush=True)
        logprobs_list.extend(generation_result.top_logprobs)
        if generation_result.stop_condition:
            print(
                f"\n\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
            )
            if generation_result.stop_condition.stop_string:
                print(f"Stop string: {generation_result.stop_condition.stop_string}")
    if args.top_logprobs:
        [print(x) for x in logprobs_list]
