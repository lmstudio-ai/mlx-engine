import math

import pytest
from pathlib import Path

from mlx_engine.cache_wrapper import (
    PROMPT_PROCESSING_CHUNK_SIZE,
    validate_prefill_step_size,
)
from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from tests.shared import model_getter, RecordingReporter
from mlx_engine.generate import load_model, create_generator, tokenize, unload

CUSTOM_PREFILL_STEP_SIZE = 256


@pytest.mark.parametrize("prefill_step_size", [None, CUSTOM_PREFILL_STEP_SIZE])
def test_validate_prefill_step_size_accepts_valid_values(prefill_step_size):
    expected_step_size = (
        PROMPT_PROCESSING_CHUNK_SIZE if prefill_step_size is None else prefill_step_size
    )
    assert validate_prefill_step_size(prefill_step_size) == expected_step_size


@pytest.mark.parametrize("prefill_step_size", [0, -1, 1.5, True])
def test_validate_prefill_step_size_rejects_invalid_values(prefill_step_size):
    with pytest.raises(
        ValueError, match="prefill_step_size must be a positive integer"
    ):
        validate_prefill_step_size(prefill_step_size)


@pytest.mark.parametrize("prefill_step_size", [0, -1, 1.5, True])
def test_load_model_rejects_invalid_prefill_step_size(prefill_step_size):
    with pytest.raises(
        ValueError, match="prefill_step_size must be a positive integer"
    ):
        load_model(model_path="unused", prefill_step_size=prefill_step_size)


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
    return math.ceil(prefillable_tokens / prefill_step_size) - 1


def _expected_sequential_prefill_updates(
    num_prompt_tokens: int, prefill_step_size: int
) -> int:
    """Compute the expected number of 'update' events for the sequential path.

    CacheWrapper._prefill drives the sequential reporter. It emits begin before
    any chunks (with prefill_tokens_processed=0), then one update per chunk,
    then finish.

    The main-model sequential path also checkpoints four tokens before the end
    of the prompt. When that checkpoint boundary lands inside a chunk, the
    chunk is split so the checkpoint can be materialized, which adds one extra
    update event.
    """
    prefillable_tokens = num_prompt_tokens - 1
    num_updates = math.ceil(prefillable_tokens / prefill_step_size)
    checkpoint_prefix_len = num_prompt_tokens - 4
    if (
        0 < checkpoint_prefix_len < prefillable_tokens
        and checkpoint_prefix_len % prefill_step_size != 0
    ):
        num_updates += 1
    return num_updates


def _expected_vision_prefill_updates(
    num_prompt_tokens: int, prefill_step_size: int
) -> int:
    """Compute the expected number of 'update' events for the VisionModelKit path.

    VisionModelKit text-only prompts go through stream_generate, which fires
    callbacks via MlxLmReporterAdapter (emit_begin=True). generate_step processes
    num_tokens - 1 tokens in chunks (the last token seeds decode), so it fires
    ceil((num_tokens - 1) / step_size) + 2 total callbacks (one initial, the
    chunked prefill, and one completion). MlxLmReporterAdapter converts the first
    into begin and the last into finish, so:
    num_updates = ceil((num_tokens - 1) / step_size).
    """
    prefillable_tokens = num_prompt_tokens - 1
    return math.ceil(prefillable_tokens / prefill_step_size)


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


@pytest.fixture
def vision_model_kit_custom_prefill():
    """Load a VisionModelKit model with a custom prefill_step_size for text-only prompts."""
    model_path = model_getter("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
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
    """Verify that batched generation honours a custom prefill_step_size."""
    model_kit = batched_model_kit_custom_prefill

    # GIVEN
    assert isinstance(model_kit, BatchedModelKit)
    prompt_tokens = _long_prompt_tokens(model_kit)
    expected_updates = _expected_batched_prefill_updates(
        len(prompt_tokens), CUSTOM_PREFILL_STEP_SIZE
    )
    expected_at_default = _expected_batched_prefill_updates(
        len(prompt_tokens), PROMPT_PROCESSING_CHUNK_SIZE
    )
    assert expected_updates != expected_at_default, (
        f"Test prompt ({len(prompt_tokens)} tokens) is not long enough to "
        f"distinguish chunk size {CUSTOM_PREFILL_STEP_SIZE} from default {PROMPT_PROCESSING_CHUNK_SIZE}"
    )

    # WHEN
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

    # THEN
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{CUSTOM_PREFILL_STEP_SIZE}, but got {len(update_events)}."
    )


def test_sequential_prefill_step_size(sequential_model_kit_custom_prefill):
    """Verify that sequential generation honours a custom prefill_step_size."""
    model_kit = sequential_model_kit_custom_prefill

    # GIVEN
    assert isinstance(model_kit, ModelKit)
    prompt_tokens = _long_prompt_tokens(model_kit)
    expected_updates = _expected_sequential_prefill_updates(
        len(prompt_tokens), CUSTOM_PREFILL_STEP_SIZE
    )
    expected_at_default = _expected_sequential_prefill_updates(
        len(prompt_tokens), PROMPT_PROCESSING_CHUNK_SIZE
    )
    assert expected_updates != expected_at_default, (
        f"Test prompt ({len(prompt_tokens)} tokens) is not long enough to "
        f"distinguish chunk size {CUSTOM_PREFILL_STEP_SIZE} from default {PROMPT_PROCESSING_CHUNK_SIZE}"
    )

    # WHEN
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

    # THEN
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{CUSTOM_PREFILL_STEP_SIZE}, but got {len(update_events)}."
    )


def test_vision_model_text_only_prefill_step_size(vision_model_kit_custom_prefill):
    """Verify that VisionModelKit text-only generation respects a custom prefill_step_size."""
    model_kit = vision_model_kit_custom_prefill

    # GIVEN
    assert isinstance(model_kit, VisionModelKit)
    prompt_tokens = _long_prompt_tokens(model_kit)
    expected_updates = _expected_vision_prefill_updates(
        len(prompt_tokens), CUSTOM_PREFILL_STEP_SIZE
    )
    expected_at_default = _expected_vision_prefill_updates(
        len(prompt_tokens), PROMPT_PROCESSING_CHUNK_SIZE
    )
    assert expected_updates != expected_at_default, (
        f"Test prompt ({len(prompt_tokens)} tokens) is not long enough to "
        f"distinguish chunk size {CUSTOM_PREFILL_STEP_SIZE} from default {PROMPT_PROCESSING_CHUNK_SIZE}"
    )

    # WHEN
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

    # THEN
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{CUSTOM_PREFILL_STEP_SIZE}, but got {len(update_events)}."
    )
