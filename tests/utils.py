from pathlib import Path
import sys
import subprocess

from mlx_engine.generate import load_model, tokenize


def model_helper(model_name: str, prompt: str, max_kv_size=4096, trust_remote_code=False, text_only=False, images_b64=None):
    """Helper method to test a model"""
    print(f"Testing model {model_name}")

    model_path_prefix = Path("~/.cache/lm-studio/models").expanduser().resolve()
    model_path = model_path_prefix / model_name

    # Check if model exists, if not prompt user to download
    if not model_path.exists():
        print(f"\nModel {model_name} not found at {model_path}")

        def greenify(text):
            return f"\033[92m{text}\033[0m"

        response = input(
            f"Would you like to download the model {greenify(model_name)}? (y/N): "
        )
        if response.lower() == "y":
            print(f"Downloading model with command: lms get {model_name}")
            subprocess.run(["lms", "get", model_name], check=True)
        else:
            print(f"Model {model_name} not found")
            sys.exit(1)

    # Load the model
    model_kit = load_model(
        model_path=model_path, max_kv_size=max_kv_size, trust_remote_code=trust_remote_code
    )

    # Tokenize the prompt
    prompt_tokens = tokenize(model_kit, prompt)

    return model_kit, prompt_tokens
