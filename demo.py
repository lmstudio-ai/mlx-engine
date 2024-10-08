import argparse

from mlx_engine.generate import load_model, create_generator, tokenize
from pathlib import Path

DEFAULT_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Explain the rules of sudoku<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LM Studio mlx-engine inference script")
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
    return parser

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    model_path = args.model
    load_model(str(model_path), max_kv_size=4096, trust_remote_code=False)

    prompt = args.prompt
    prompt_tokens = tokenize(prompt)
    generator = create_generator(prompt_tokens, None, None, {"max_tokens": 1024})
    for token in generator:
        print(token, end="", flush=True)
    print()
