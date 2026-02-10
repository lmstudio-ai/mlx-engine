from pathlib import Path
import sys
import subprocess
from typing import Optional

from mlx_engine.generate import load_model, load_draft_model, tokenize
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter


class RecordingReporter(PromptProgressReporter):
    """A reporter that records each event it receives, for testing."""

    def __init__(self):
        self.events: list[dict] = []

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        event = {
            "type": "begin",
            "is_draft": is_draft,
            "cached_tokens": cached_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "prefill_tokens_processed": prefill_tokens_processed,
        }
        self.events.append(event)
        print(event)
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        event = {
            "type": "update",
            "is_draft": is_draft,
            "prefill_tokens_processed": prefill_tokens_processed,
        }
        self.events.append(event)
        print(event)
        return True

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        event = {
            "type": "finish",
            "is_draft": is_draft,
            "prefill_tokens_processed": prefill_tokens_processed,
        }
        self.events.append(event)
        print(event)
        return True


class CancellingReporter(RecordingReporter):
    """A reporter that cancels after a specified number of events, for testing."""

    def __init__(self, cancel_after: int):
        super().__init__()
        self.cancel_after = cancel_after

    def _should_continue(self) -> bool:
        return len(self.events) < self.cancel_after

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        super().begin(
            is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed
        )
        return self._should_continue()

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        super().update(is_draft, prefill_tokens_processed)
        return self._should_continue()

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        super().finish(is_draft, prefill_tokens_processed)
        return self._should_continue()


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
    max_num_seqs=None,
):
    """Helper method to test a model"""
    print(f"Testing model {model_name}")

    # Check if model exists, if not prompt user to download
    model_path = model_getter(model_name)

    # Load the model
    model_kit = load_model(
        model_path=model_path,
        max_kv_size=max_kv_size,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
    )

    # Load the draft model if any
    if draft_model_name is not None:
        draft_model_path = model_getter(draft_model_name)
        load_draft_model(model_kit, draft_model_path)

    # Tokenize the prompt
    prompt_tokens = tokenize(model_kit, prompt)

    return model_kit, prompt_tokens
