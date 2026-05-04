from queue import Queue
from types import SimpleNamespace

import mlx.core as mx

from mlx_engine.model_kit.batched_model_kit_types import RequestCancelled
from mlx_engine.model_kit.batched_vision.request_lifecycle import (
    ActiveRequest,
    FailedRestore,
    GenerationRequest,
    GenerationThreadController,
    GenerationThreadState,
    PreparedInsert,
)


def _sampler(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        rqueue=Queue(),
        prompt_tokens=[1, 2, 3],
        request_id=request_id,
        images_b64=None,
        sampler=_sampler,
        logits_processors=[],
        top_logprobs=0,
        max_tokens=1,
    )


def _prepared(request: GenerationRequest) -> PreparedInsert:
    return PreparedInsert(
        request=request,
        prepared_prompt=object(),
        restored=None,
    )


class _FakeBatchGenerator:
    def __init__(self, next_results=None):
        self.next_results = list(next_results or [])
        self.removed = []
        self.closed = False

    def next(self):
        return self.next_results.pop(0)

    def remove(self, uid):
        self.removed.append(uid)
        return True

    def close(self):
        self.closed = True


def _active_request(request_id: str = "request") -> ActiveRequest:
    return ActiveRequest(
        rqueue=Queue(),
        detokenizer=object(),
        top_logprobs=0,
        request_id=request_id,
        image_spans=[],
    )


def _controller(
    state: GenerationThreadState,
    *,
    max_seq_nums: int = 4,
    enqueue_restore=None,
    insert_prepared_request=None,
    emit_response=None,
    finish_response=None,
) -> GenerationThreadController:
    return GenerationThreadController(
        state=state,
        request_queue=Queue(),
        max_seq_nums=max_seq_nums,
        enqueue_restore=enqueue_restore or (lambda _request: None),
        insert_prepared_request=insert_prepared_request
        or (lambda _batch_generator, _prepared_insert, _active: None),
        emit_response=emit_response or (lambda _active_request, response: response),
        finish_response=finish_response
        or (lambda _active_request, _response, _keep_hot_cache: None),
    )


def test_request_lifecycle_admits_requests_up_to_capacity():
    """Pending requests reserve restore slots without exceeding max sequences."""
    restore_calls = []
    first = _request("first")
    second = _request("second")
    state = GenerationThreadState(batch_generator=_FakeBatchGenerator())
    state.active[0] = _active_request("active")
    state.pending = [first, second]
    controller = _controller(
        state,
        max_seq_nums=2,
        enqueue_restore=restore_calls.append,
    )

    controller.admit_pending_requests()

    assert restore_calls == [first]
    assert state.restoring == {"first": first}
    assert state.pending == [second]


def test_request_lifecycle_inserts_ready_requests_up_to_capacity():
    """Ready inserts consume available active slots and leave the rest queued."""
    first = _prepared(_request("first"))
    second = _prepared(_request("second"))
    state = GenerationThreadState(batch_generator=_FakeBatchGenerator())
    state.active[0] = _active_request("active")
    state.ready = [first, second]
    inserted = []

    def insert_prepared(_batch_generator, prepared_insert, active):
        inserted.append(prepared_insert)
        active[len(active)] = _active_request(prepared_insert.request.request_id)

    controller = _controller(
        state,
        max_seq_nums=2,
        insert_prepared_request=insert_prepared,
    )

    controller.insert_ready_requests()

    assert inserted == [first]
    assert state.ready == [second]
    assert len(state.active) == 2


def test_request_lifecycle_cancel_restoring_ignores_late_restore():
    """Cancelling an in-flight restore prevents a late prepared insert."""
    request = _request("request")
    state = GenerationThreadState(batch_generator=_FakeBatchGenerator())
    state.restoring[request.request_id] = request
    controller = _controller(state)

    assert controller.cancel_request(request.request_id)
    controller.handle_prepared_event(_prepared(request))

    assert isinstance(request.rqueue.get_nowait(), RequestCancelled)
    assert state.ready == []
    assert state.restoring == {}
    assert state.cancelled_restores == set()


def test_request_lifecycle_failed_restore_reaches_request_queue():
    """Restore errors are delivered to the request instead of becoming ready."""
    request = _request("request")
    error = RuntimeError("restore failed")
    state = GenerationThreadState(batch_generator=_FakeBatchGenerator())
    state.restoring[request.request_id] = request
    controller = _controller(state)

    controller.handle_prepared_event(FailedRestore(request, error))

    assert request.rqueue.get_nowait() is error
    assert state.ready == []
    assert state.restoring == {}


def test_request_lifecycle_steps_generation_and_stores_finished_hot_cache():
    """Finished responses emit tokens, close the queue, and run cache cleanup."""
    request_state = _active_request("request")
    prompt_response = SimpleNamespace(uid=5, progress=(8, 5))
    generation_response = SimpleNamespace(
        uid=5,
        token=42,
        token_logprob=-0.25,
        top_logprobs=None,
        finish_reason="length",
        prompt_cache=["cache"],
        all_tokens=[1, 2, 42],
        rope_deltas="rope",
    )
    batch_generator = _FakeBatchGenerator(
        next_results=[([prompt_response], [generation_response])]
    )
    state = GenerationThreadState(batch_generator=batch_generator)
    state.active[5] = request_state
    finished = []
    controller = _controller(
        state,
        emit_response=lambda _request_state, response: f"token:{response.token}",
        finish_response=lambda active, response, _keep_hot_cache: finished.append(
            (active, response)
        ),
    )

    controller.step_generation()

    assert request_state.rqueue.get_nowait() == (5, 5)
    assert request_state.rqueue.get_nowait() == "token:42"
    assert request_state.rqueue.get_nowait() is None
    assert finished == [(request_state, generation_response)]
    assert state.active == {}
