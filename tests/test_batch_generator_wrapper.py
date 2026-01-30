import pytest

from mlx_engine.generate import (
    MAX_TOP_LOGPROBS,
    create_generator,
    load_model,
    tokenize,
)
from mlx_engine.batch_generate import create_batch_generator
from tests.shared import model_getter

MODEL_NAME = "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
PROMPT_TEXT = """<|im_start|>user
Write one short sentence about continuous batching.
<|im_end|>\n<|im_start|>assistant\n"""
SECOND_PROMPT_TEXT = """<|im_start|>user
Name two benefits of continuous batching.
<|im_end|>\n<|im_start|>assistant\n"""
ASYNC_PROMPT_TEXT = """<|im_start|>user
Write a detailed, multi-paragraph explanation of continuous batching.
<|im_end|>\n<|im_start|>assistant\n"""
DEFAULT_MAX_TOKENS = 512
MAX_ITERATIONS = DEFAULT_MAX_TOKENS * 4
ASYNC_WAIT_CYCLES = 3
VERBOSE = False


@pytest.fixture(scope="module")
def model_kit():
    model_path = model_getter(MODEL_NAME)
    return load_model(model_path=model_path, max_kv_size=4096)


def _collect_request_tokens(
    batch_generator,
    request_slot_id: int,
    max_iterations: int = MAX_ITERATIONS,
) -> list[int]:
    collected_tokens: list[int] = []
    stop_emitted = False
    iteration_count = 0
    while stop_emitted is False and iteration_count < max_iterations:
        batch_results = batch_generator.next()
        iteration_count += 1
        if len(batch_results) == 0:
            continue
        for result in batch_results:
            if result.request_slot_id != request_slot_id:
                continue
            _print_batch_text(result.request_slot_id, result.text)
            for token in result.tokens:
                collected_tokens.append(token.id)
            if result.stop_condition is not None:
                stop_emitted = True
                break
    if stop_emitted is False:
        raise AssertionError("Batch generator did not stop within the expected iterations")
    return collected_tokens


def _drain_request(
    batch_generator,
    request_slot_id: int,
    max_iterations: int = MAX_ITERATIONS,
) -> None:
    stop_emitted = False
    iteration_count = 0
    while stop_emitted is False and iteration_count < max_iterations:
        batch_results = batch_generator.next()
        iteration_count += 1
        if len(batch_results) == 0:
            continue
        for result in batch_results:
            if result.request_slot_id != request_slot_id:
                continue
            _print_batch_text(result.request_slot_id, result.text)
            if result.stop_condition is not None:
                stop_emitted = True
                break
    if stop_emitted is False:
        raise AssertionError("Batch generator did not stop within the expected iterations")


def _collect_parallel_results(
    batch_generator,
    request_slot_ids: list[int],
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[dict[int, list[int]], dict[int, object], bool]:
    tokens_by_request: dict[int, list[int]] = {
        request_slot_id: [] for request_slot_id in request_slot_ids
    }
    stop_conditions_by_request: dict[int, object] = {}
    remaining_request_slot_ids = set(request_slot_ids)
    iteration_count = 0
    saw_interleaved = False

    while len(remaining_request_slot_ids) > 0 and iteration_count < max_iterations:
        batch_results = batch_generator.next()
        iteration_count += 1
        if len(batch_results) == 0:
            continue
        batch_request_slot_ids = set()
        for result in batch_results:
            request_slot_id = result.request_slot_id
            if request_slot_id not in tokens_by_request:
                continue
            batch_request_slot_ids.add(request_slot_id)
            _print_batch_text(request_slot_id, result.text)
            for token in result.tokens:
                tokens_by_request[request_slot_id].append(token.id)
            if result.stop_condition is not None:
                stop_conditions_by_request[request_slot_id] = result.stop_condition
                if request_slot_id in remaining_request_slot_ids:
                    remaining_request_slot_ids.remove(request_slot_id)
        if len(batch_request_slot_ids) > 1:
            saw_interleaved = True

    if len(remaining_request_slot_ids) > 0:
        raise AssertionError("Batch generator did not finish all requests in time")

    return tokens_by_request, stop_conditions_by_request, saw_interleaved


def _print_generation_text(text: str) -> None:
    if not VERBOSE or text == "":
        return
    print(text, end="", flush=True)


def _print_batch_text(request_slot_id: int, text: str) -> None:
    if not VERBOSE or text == "":
        return
    print(f"[request {request_slot_id}] {text}", end="", flush=True)


def test_create_generator_smoke_text_model(model_kit):
    prompt_tokens = tokenize(model_kit, PROMPT_TEXT)
    generated_text = ""
    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prompt_tokens,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=0,
        temp=0.0,
    ):
        generated_text += result.text
        _print_generation_text(result.text)
        if result.stop_condition is not None:
            break
    assert len(generated_text) > 0


def test_batch_generator_wrapper_caches_prompt_per_session(model_kit):
    prompt_tokens = tokenize(model_kit, PROMPT_TEXT)
    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=8,
    )
    try:
        (
            first_request_slot_id,
            cached_tokens,
            total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="session-a",
            prompt_tokens=prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert cached_tokens == 0
        assert total_prompt_tokens == len(prompt_tokens)

        generated_tokens = _collect_request_tokens(
            batch_generator, first_request_slot_id
        )
        assert len(generated_tokens) > 0

        extended_prompt_tokens = prompt_tokens + generated_tokens
        (
            second_request_slot_id,
            cached_tokens,
            total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="session-a",
            prompt_tokens=extended_prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert cached_tokens > 0
        assert cached_tokens < len(extended_prompt_tokens)
        assert total_prompt_tokens == len(extended_prompt_tokens)
        _drain_request(batch_generator, second_request_slot_id)

        (
            third_request_slot_id,
            cached_tokens,
            total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="session-b",
            prompt_tokens=extended_prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert cached_tokens == 0
        assert total_prompt_tokens == len(extended_prompt_tokens)
        _drain_request(batch_generator, third_request_slot_id)
    finally:
        batch_generator.close()


def test_batch_generator_parallel_generation(model_kit):
    prompt_tokens = tokenize(model_kit, PROMPT_TEXT)
    second_prompt_tokens = tokenize(model_kit, SECOND_PROMPT_TEXT)
    prompt_progress_events: list[tuple[int, int, int]] = []

    def prompt_progress_callback(events: list[tuple[int, int, int]]) -> None:
        prompt_progress_events.extend(events)

    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=8,
        prompt_progress_callback=prompt_progress_callback,
    )
    try:
        (
            first_request_slot_id,
            first_cached_tokens,
            first_total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="parallel-session-a",
            prompt_tokens=prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        (
            second_request_slot_id,
            second_cached_tokens,
            second_total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="parallel-session-b",
            prompt_tokens=second_prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        assert first_request_slot_id != second_request_slot_id
        assert first_cached_tokens == 0
        assert second_cached_tokens == 0
        assert first_total_prompt_tokens == len(prompt_tokens)
        assert second_total_prompt_tokens == len(second_prompt_tokens)

        tokens_by_request, stop_conditions_by_request, saw_interleaved = _collect_parallel_results(
            batch_generator, [first_request_slot_id, second_request_slot_id]
        )
        assert len(tokens_by_request[first_request_slot_id]) > 0
        assert len(tokens_by_request[second_request_slot_id]) > 0
        assert first_request_slot_id in stop_conditions_by_request
        assert second_request_slot_id in stop_conditions_by_request
        assert saw_interleaved is True

        assert len(prompt_progress_events) > 0
        for request_slot_id, processed_tokens, total_tokens in prompt_progress_events:
            assert request_slot_id in (first_request_slot_id, second_request_slot_id)
            assert processed_tokens <= total_tokens
    finally:
        batch_generator.close()


def test_batch_generator_wrapper_rejects_top_logprobs_over_max(model_kit):
    prompt_tokens = tokenize(model_kit, "Hello from the batch generator.")
    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=8,
    )
    try:
        with pytest.raises(ValueError):
            batch_generator.insert(
                session_id="session-a",
                prompt_tokens=prompt_tokens,
                max_tokens=DEFAULT_MAX_TOKENS,
                top_logprobs=MAX_TOP_LOGPROBS + 1,
            )
    finally:
        batch_generator.close()


def test_batch_generator_returns_top_logprobs(model_kit):
    prompt_tokens = tokenize(model_kit, "Return a short response.")
    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=8,
    )
    try:
        (
            request_slot_id,
            cached_tokens,
            total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="top-logprobs-session",
            prompt_tokens=prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
            top_logprobs=1,
        )
        assert cached_tokens == 0
        assert total_prompt_tokens == len(prompt_tokens)

        top_logprobs_entries: list[list] = []
        stop_emitted = False
        iteration_count = 0
        while stop_emitted is False and iteration_count < MAX_ITERATIONS:
            batch_results = batch_generator.next()
            iteration_count += 1
            if len(batch_results) == 0:
                continue
            for result in batch_results:
                if result.request_slot_id != request_slot_id:
                    continue
                _print_batch_text(result.request_slot_id, result.text)
                if len(result.top_logprobs) > 0:
                    top_logprobs_entries.extend(result.top_logprobs)
                if result.stop_condition is not None:
                    stop_emitted = True
                    break

        assert stop_emitted is True
        assert len(top_logprobs_entries) > 0
        for entry in top_logprobs_entries:
            assert len(entry) == 1
    finally:
        batch_generator.close()


def test_batch_generator_async_insert(model_kit):
    prompt_tokens = tokenize(model_kit, ASYNC_PROMPT_TEXT)
    second_prompt_tokens = tokenize(model_kit, SECOND_PROMPT_TEXT)
    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=2,
        prefill_batch_size=1,
        prefill_step_size=8,
    )
    try:
        (
            first_request_slot_id,
            first_cached_tokens,
            first_total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="async-session-a",
            prompt_tokens=prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert first_cached_tokens == 0
        assert first_total_prompt_tokens == len(prompt_tokens)

        cycle_count = 0
        stop_before_insert = False
        while cycle_count < ASYNC_WAIT_CYCLES and stop_before_insert is False:
            batch_results = batch_generator.next()
            cycle_count += 1
            if len(batch_results) == 0:
                continue
            for result in batch_results:
                if result.request_slot_id != first_request_slot_id:
                    continue
                _print_batch_text(result.request_slot_id, result.text)
                if result.stop_condition is not None:
                    stop_before_insert = True
                    break

        if stop_before_insert is True:
            raise AssertionError(
                "Async insert test ended early; reduce ASYNC_WAIT_CYCLES or adjust ASYNC_PROMPT_TEXT."
            )
        
        (
            second_request_slot_id,
            second_cached_tokens,
            second_total_prompt_tokens,
        ) = batch_generator.insert(
            session_id="async-session-b",
            prompt_tokens=second_prompt_tokens,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert second_cached_tokens == 0
        assert second_total_prompt_tokens == len(second_prompt_tokens)

        tokens_by_request, stop_conditions_by_request, saw_interleaved = _collect_parallel_results(
            batch_generator, [first_request_slot_id, second_request_slot_id]
        )
        assert len(tokens_by_request[first_request_slot_id]) > 0
        assert len(tokens_by_request[second_request_slot_id]) > 0
        assert first_request_slot_id in stop_conditions_by_request
        assert second_request_slot_id in stop_conditions_by_request
        assert saw_interleaved is True
    finally:
        batch_generator.close()
