import pytest
import threading
import time

from tests.shared import model_getter
from mlx_engine.generate import load_model, create_generator, tokenize, unload


@pytest.fixture
def model_kit():
    """Load model once for all tests."""
    model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
    kit = load_model(model_path=model_path, max_kv_size=4096, seed=0)
    yield kit
    unload(kit)


def test_batched_generation_two_threads(model_kit):
    """Test batched generation with two concurrent threads."""

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
