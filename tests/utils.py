from pathlib import Path
import sys
import subprocess

from mlx_engine.generate import load_model, load_draft_model, tokenize


def model_getter(model_name: str):
    """Helper method to get a model, prompt user to download if not found"""

    with open(Path("~/.lmstudio-home-pointer").expanduser().resolve(), "r") as f:
        lmstudio_home = Path(f.read().strip())
    model_path = lmstudio_home / "models" / model_name

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

    return model_path


def model_load_and_tokenize_prompt(
    model_name: str,
    prompt: str,
    max_kv_size=4096,
    trust_remote_code=False,
    draft_model_name=None,
):
    """Helper method to test a model"""
    print(f"Testing model {model_name}")

    # Check if model exists, if not prompt user to download
    model_path = model_getter(model_name)

    # Load the model
    model_kit = load_model(
        model_path=model_path,
        max_kv_size=max_kv_size,
        trust_remote_code=trust_remote_code,
    )

    # Load the draft model if any
    if draft_model_name is not None:
        draft_model_path = model_getter(draft_model_name)
        load_draft_model(model_kit, draft_model_path)

    # Tokenize the prompt
    prompt_tokens = tokenize(model_kit, prompt)

    return model_kit, prompt_tokens


def read_text_file(file_path: str) -> str:
    """
    Read content from a text file with error handling.

    Args:
        file_path: Path to the text file to read

    Returns:
        The content of the file as a string

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For other reading errors
    """
    try:
        # Convert string path to Path object for better path handling
        path = Path(file_path).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
