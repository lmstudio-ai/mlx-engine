import math

import pytest
import threading
import time
from pathlib import Path

from mlx_engine.cache_wrapper import PROMPT_PROCESSING_CHUNK_SIZE
from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from tests.shared import model_getter, RecordingReporter
from mlx_engine.generate import load_model, create_generator, tokenize, unload

# The default prefill_step_size in mlx_lm.generate.BatchGenerator
MLX_LM_DEFAULT_PREFILL_STEP_SIZE = 2048


def _expected_batched_prefill_updates(num_prompt_tokens: int) -> int:
    """Compute the expected number of 'update' events from BatchedMlxLmReporterAdapter.

    BatchGenerator._process_prompts fires one progress callback per prefill chunk,
    leaving prompt_checkpoint (1) token for the final step.
    BatchedMlxLmReporterAdapter maps these callbacks to events: the first callback
    emits both begin and update (no early return after begin), middle callbacks emit
    update, and the last emits finish. So: num_updates = num_callbacks - 1.
    """
    prefillable_tokens = num_prompt_tokens - 1

    expected_at_correct_size = (
        math.ceil(prefillable_tokens / PROMPT_PROCESSING_CHUNK_SIZE) - 1
    )
    expected_at_default_size = (
        math.ceil(prefillable_tokens / MLX_LM_DEFAULT_PREFILL_STEP_SIZE) - 1
    )

    assert expected_at_correct_size != expected_at_default_size, (
        f"Test prompt ({num_prompt_tokens} tokens) is not long enough to "
        f"distinguish chunk size {PROMPT_PROCESSING_CHUNK_SIZE} from {MLX_LM_DEFAULT_PREFILL_STEP_SIZE}"
    )

    return expected_at_correct_size


@pytest.fixture
def model_kit():
    """Load model once for all tests."""
    model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
    kit = load_model(model_path=model_path, max_kv_size=4096, seed=0)
    yield kit
    unload(kit)


def test_batched_generation_max_tokens(model_kit):
    """Test that batched generation stops with token_limit when max_tokens is reached."""

    assert isinstance(model_kit, BatchedModelKit)

    prompt = """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
"""
    prompt_tokens = tokenize(model_kit, prompt)

    max_tokens = 5
    token_count = 0
    stop_condition = None

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        seed=0,
        max_tokens=max_tokens,
        temp=0.0,
    ):
        token_count += len(result.tokens)
        if result.stop_condition:
            stop_condition = result.stop_condition
            break

    assert stop_condition is not None
    assert stop_condition.stop_reason == "token_limit"
    assert token_count <= max_tokens


@pytest.fixture
def model_kit_large_kv():
    """Load model with a large KV cache for long-prompt tests."""
    model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
    kit = load_model(model_path=model_path, max_kv_size=20000, seed=0)
    yield kit
    unload(kit)


def test_batched_prefill_step_size(model_kit_large_kv):
    """Verify that batched generation uses PROMPT_PROCESSING_CHUNK_SIZE for prefill.

    A prompt longer than PROMPT_PROCESSING_CHUNK_SIZE (512) should produce
    multiple progress update events. If the batch path silently uses a
    larger chunk size (e.g. the mlx-lm default of 2048), the prompt would
    be processed in fewer steps, producing fewer update events.
    """
    assert isinstance(model_kit_large_kv, BatchedModelKit)

    test_data_dir = Path(__file__).parent / "data"
    file_content = (test_data_dir / "ben_franklin_autobiography_start.txt").read_text()
    prompt = f"""<|im_start|>user
```
{file_content}
```
Who is this passage about? Only say the name, and nothing else<|im_end|>
<|im_start|>assistant
"""
    prompt_tokens = tokenize(model_kit_large_kv, prompt)

    # The prompt must be long enough that different chunk sizes produce
    # visibly different callback patterns
    assert len(prompt_tokens) > PROMPT_PROCESSING_CHUNK_SIZE * 2, (
        f"Prompt has {len(prompt_tokens)} tokens, need >"
        f" {PROMPT_PROCESSING_CHUNK_SIZE * 2} to distinguish chunk sizes"
    )

    reporter = RecordingReporter()

    for result in create_generator(
        model_kit=model_kit_large_kv,
        prompt_tokens=prompt_tokens,
        seed=0,
        max_tokens=5,
        temp=0.0,
        prompt_progress_reporter=reporter,
    ):
        if result.stop_condition:
            break

    expected_updates = _expected_batched_prefill_updates(len(prompt_tokens))
    update_events = [event for event in reporter.events if event["type"] == "update"]
    assert len(update_events) == expected_updates, (
        f"Expected {expected_updates} prefill progress updates "
        f"for {len(prompt_tokens)} tokens with chunk size "
        f"{PROMPT_PROCESSING_CHUNK_SIZE}, but got {len(update_events)}."
    )


def test_batched_generation_two_threads(model_kit):
    """Test batched generation with two concurrent threads."""

    assert isinstance(model_kit, BatchedModelKit)

    # Define two different prompts with different topics
    prompts = [
        """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
""",
        """<|im_start|>user
Explain how photosynthesis works in plants.<|im_end|>
<|im_start|>assistant
""",
    ]

    # Tokenize prompts
    prompt_tokens_list = [tokenize(model_kit, prompt) for prompt in prompts]

    # Storage for results from each thread
    results = {}

    def run_generation(thread_id: int, prompt_tokens: list):
        """Run generation in a thread and store the result."""
        # Use the thread name as the request_id
        request_id = str(thread_id)
        generated_text = ""

        for result in create_generator(
            model_kit=model_kit,
            prompt_tokens=prompt_tokens,
            request_id=request_id,
            seed=0,
            max_tokens=100,
            temp=0.0,
        ):
            generated_text += result.text
            if result.stop_condition:
                break

        results[thread_id] = generated_text

    # Create threads
    threads = [
        threading.Thread(
            target=run_generation,
            args=(i + 1, prompt_tokens),
        )
        for i, prompt_tokens in enumerate(prompt_tokens_list)
    ]

    # Measure wall time for concurrent execution
    start_time = time.perf_counter()

    for thread in threads:
        thread.start()

    # Wait for threads with timeout
    for thread in threads:
        thread.join(timeout=20.0)

    end_time = time.perf_counter()
    wall_time = end_time - start_time

    # Verify all threads completed
    for i, thread in enumerate(threads, 1):
        assert not thread.is_alive()
        assert i in results
        assert len(results[i]) > 0

    # Assert relevance to prompts
    text1, text2 = results[1].lower(), results[2].lower()
    assert "paris" in text1 and "paris" not in text2
    assert "chlorophyll" in text2 and "chlorophyll" not in text1

    # Print results
    print(f"\nWall time: {wall_time:.3f} seconds")
    print(f"\nThread 1 output:\n{results[1]}")
    print(f"\nThread 2 output:\n{results[2]}")
