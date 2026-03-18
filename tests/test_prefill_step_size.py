import math

import pytest
from pathlib import Path

from mlx_engine.cache_wrapper import PROMPT_PROCESSING_CHUNK_SIZE
from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from mlx_engine.model_kit.model_kit import ModelKit
from tests.shared import model_getter, RecordingReporter
from mlx_engine.generate import load_model, create_generator, tokenize, unload

CUSTOM_PREFILL_STEP_SIZE = 256


def _expected_batched_prefill_updates(
    num_prompt_tokens: int, prefill_step_size: int
) -> int:
    """Compute the expected number of 'update' events from BatchedMlxLmReporterAdapter.

    BatchGenerator._process_prompts fires one progress callback per prefill chunk,
    leaving prompt_checkpoint (1) token for the final step.
    BatchedMlxLmReporterAdapter maps these callbacks to events: the first callback
    emits both begin and update (no early return after begin), middle callbacks emit
    update, and the last emits finish. So: num_updates = num_callbacks - 1.
    """
    prefillable_tokens = num_prompt_tokens - 1

    expected_at_requested_size = math.ceil(prefillable_tokens / prefill_step_size) - 1
    expected_at_default_size = (
        math.ceil(prefillable_tokens / PROMPT_PROCESSING_CHUNK_SIZE) - 1
    )

    assert expected_at_requested_size != expected_at_default_size, (
        f"Test prompt ({num_prompt_tokens} tokens) is not long enough to "
        f"distinguish chunk size {prefill_step_size} from default {PROMPT_PROCESSING_CHUNK_SIZE}"
    )

    return expected_at_requested_size


def _expected_sequential_prefill_updates(
    num_prompt_tokens: int, prefill_step_size: int
) -> int:
    """Compute the expected number of 'update' events for the sequential path.

    CacheWrapper._prefill drives the sequential reporter. It emits begin before
    any chunks (with prefill_tokens_processed=0), then one update per chunk,
    then finish. So: num_updates = ceil((num_tokens - 1) / prefill_step_size).
    """
    prefillable_tokens = num_prompt_tokens - 1

    expected_at_requested_size = math.ceil(prefillable_tokens / prefill_step_size)
    expected_at_default_size = math.ceil(
        prefillable_tokens / PROMPT_PROCESSING_CHUNK_SIZE
    )

    assert expected_at_requested_size != expected_at_default_size, (
        f"Test prompt ({num_prompt_tokens} tokens) is not long enough to "
        f"distinguish chunk size {prefill_step_size} from default {PROMPT_PROCESSING_CHUNK_SIZE}"
    )

    return expected_at_requested_size


def _long_prompt_tokens(model_kit):
    """Build a long prompt (~7k tokens) and return the token list."""
    test_data_dir = Path(__file__).parent / "data"
    file_content = (test_data_dir / "ben_franklin_autobiography_start.txt").read_text()
    prompt = f"""<|im_start|>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<|im_end|>
<|im_start|>assistant
"""
    return tokenize(model_kit, prompt)


@pytest.fixture
def batched_model_kit_custom_prefill():
    """Load batched model with a custom prefill_step_size and large KV cache."""
    model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
    kit = load_model(
        model_path=model_path,
        max_kv_size=20000,
        seed=0,
        prefill_step_size=CUSTOM_PREFILL_STEP_SIZE,
    )
    yield kit
    unload(kit)


@pytest.fixture
def sequential_model_kit_custom_prefill():
    """Load sequential model with a custom prefill_step_size and large KV cache."""
    model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
    kit = load_model(
        model_path=model_path,
        max_kv_size=20000,
        max_seq_nums=1,
        seed=0,
        prefill_step_size=CUSTOM_PREFILL_STEP_SIZE,
    )
    yield kit
    unload(kit)


def test_batched_prefill_step_size(batched_model_kit_custom_prefill):
    """Verify that batched generation honours a custom prefill_step_size.

    Loads with prefill_step_size=256 and sends a long prompt. The number of
    progress update events must match 256-token chunks, not the default 512.
    """
    model_kit = batched_model_kit_custom_prefill
    assert isinstance(model_kit, BatchedModelKit)

    prompt_tokens = _long_prompt_tokens(model_kit)
    reporter = RecordingReporter()

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        seed=0,
        max_tokens=5,
        temp=0.0,
        prompt_progress_reporter=reporter,
    ):
        if result.stop_condition:
            break

    expected_updates = _expected_batched_prefill_updates(
        len(prompt_tokens), CUSTOM_PREFILL_STEP_SIZE
    )
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{CUSTOM_PREFILL_STEP_SIZE}, but got {len(update_events)}."
    )


def test_sequential_prefill_step_size(sequential_model_kit_custom_prefill):
    """Verify that sequential generation honours a custom prefill_step_size.

    Loads with prefill_step_size=256 and max_seq_nums=1 (forcing the sequential
    path). The number of progress update events must match 256-token chunks,
    not the default 512.
    """
    model_kit = sequential_model_kit_custom_prefill
    assert isinstance(model_kit, ModelKit)

    prompt_tokens = _long_prompt_tokens(model_kit)
    reporter = RecordingReporter()

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        seed=0,
        max_tokens=5,
        temp=0.0,
        prompt_progress_reporter=reporter,
    ):
        if result.stop_condition:
            break

    expected_updates = _expected_sequential_prefill_updates(
        len(prompt_tokens), CUSTOM_PREFILL_STEP_SIZE
    )
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{CUSTOM_PREFILL_STEP_SIZE}, but got {len(update_events)}."
    )
