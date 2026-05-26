import logging
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Any, Callable

import mlx.core as mx
from mlx_engine.model_kit.batched_model_kit_types import (
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator as LocalVlmBatchGenerator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    RestoredPromptCache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.batched_vision.prompt_inputs import PreparedPrompt
from mlx_engine.utils.prompt_progress_events import PromptProgressEvent

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    rqueue: Queue
    prompt_tokens: list[int]
    request_id: str
    images_b64: list[str] | None
    sampler: Callable[[mx.array], mx.array]
    logits_processors: list
    top_logprobs: int
    max_tokens: int


@dataclass
class PreparedInsert:
    request: GenerationRequest
    prepared_prompt: PreparedPrompt
    restored: RestoredPromptCache | None


@dataclass
class FailedRestore:
    request: GenerationRequest
    error: Exception


@dataclass
class ActiveRequest:
    rqueue: Queue
    detokenizer: Any
    top_logprobs: int
    request_id: str
    image_spans: list[PromptImageSpan]
    cached_tokens: int
    prompt_progress_finished: bool = False


@dataclass
class GenerationThreadState:
    batch_generator: LocalVlmBatchGenerator
    active: dict[int, ActiveRequest] = field(default_factory=dict)
    pending: list[GenerationRequest] = field(default_factory=list)
    ready: list[PreparedInsert] = field(default_factory=list)
    restoring: dict[str, GenerationRequest] = field(default_factory=dict)
    cancelled_restores: set[str] = field(default_factory=set)


class GenerationThreadController:
    """Owns request lifecycle state for the VLM generation thread."""

    def __init__(
        self,
        *,
        state: GenerationThreadState,
        request_queue: Queue,
        max_seq_nums: int,
        enqueue_restore: Callable[[GenerationRequest], None],
        insert_prepared_request: Callable[
            [LocalVlmBatchGenerator, PreparedInsert, dict[int, ActiveRequest]], None
        ],
        emit_response: Callable[[ActiveRequest, Any], Any],
        finish_response: Callable[[ActiveRequest, Any, bool], None],
    ):
        self.state = state
        self._request_queue = request_queue
        self._max_seq_nums = max_seq_nums
        self._enqueue_restore = enqueue_restore
        self._insert_prepared_request = insert_prepared_request
        self._emit_response = emit_response
        self._finish_response = finish_response

    @staticmethod
    def drain_queue(queue: Queue, timeout: float | None) -> list:
        items = []
        try:
            item = (
                queue.get(timeout=timeout)
                if timeout is not None
                else queue.get_nowait()
            )
            items.append(item)
        except QueueEmpty:
            return items

        while True:
            try:
                items.append(queue.get_nowait())
            except QueueEmpty:
                return items

    def cancel_request(self, request_id: str) -> bool:
        state = self.state
        for i, request in enumerate(state.pending):
            if request.request_id == request_id:
                state.pending.pop(i)
                request.rqueue.put(RequestCancelled())
                return True

        for i, prepared_insert in enumerate(state.ready):
            if prepared_insert.request.request_id == request_id:
                state.ready.pop(i)
                prepared_insert.request.rqueue.put(RequestCancelled())
                return True

        request = state.restoring.pop(request_id, None)
        if request is not None:
            state.cancelled_restores.add(request_id)
            request.rqueue.put(RequestCancelled())
            return True

        for uid, result in list(state.active.items()):
            if result.request_id != request_id:
                continue
            state.batch_generator.remove(uid)
            result.rqueue.put(RequestCancelled())
            del state.active[uid]
            return True

        return False

    def handle_prepared_event(self, item: PreparedInsert | FailedRestore) -> None:
        request_id = item.request.request_id
        self.state.restoring.pop(request_id, None)

        if request_id in self.state.cancelled_restores:
            self.state.cancelled_restores.discard(request_id)
            return

        if isinstance(item, FailedRestore):
            item.request.rqueue.put(item.error)
            return

        self.state.ready.append(item)

    def drain_generation_events(self, timeout: float | None) -> None:
        for item in self.drain_queue(self._request_queue, timeout):
            if isinstance(item, CancelGenerationRequest):
                if not self.cancel_request(item.request_id):
                    logger.warning(f"Could not cancel request_id={item.request_id}")
                continue

            if isinstance(item, (PreparedInsert, FailedRestore)):
                self.handle_prepared_event(item)
                continue

            self.state.pending.append(item)

    def reserved_slots(self) -> int:
        state = self.state
        return len(state.active) + len(state.ready) + len(state.restoring)

    def insert_ready_requests(self) -> None:
        state = self.state
        if len(state.active) >= self._max_seq_nums:
            return

        next_ready = []
        for prepared_insert in state.ready:
            if len(state.active) < self._max_seq_nums:
                self._insert_prepared_request(
                    state.batch_generator,
                    prepared_insert,
                    state.active,
                )
            else:
                next_ready.append(prepared_insert)
        state.ready = next_ready

    def admit_pending_requests(self) -> None:
        state = self.state
        if self.reserved_slots() >= self._max_seq_nums:
            return

        next_pending = []
        for request in state.pending:
            if self.reserved_slots() < self._max_seq_nums:
                state.restoring[request.request_id] = request
                self._enqueue_restore(request)
            else:
                next_pending.append(request)
        state.pending = next_pending

    def step_generation(self) -> None:
        state = self.state
        if not state.active:
            return

        prompt_responses, generation_responses = state.batch_generator.next()
        for response in prompt_responses:
            result = state.active.get(response.uid)
            if result is None or result.prompt_progress_finished:
                continue
            processed, total = response.progress
            total_prompt_tokens = max(0, total - 1)
            processed = min(processed, total_prompt_tokens)
            prefill_tokens_processed = max(0, processed - result.cached_tokens)
            is_final = processed >= total_prompt_tokens
            if is_final:
                result.prompt_progress_finished = True
            result.rqueue.put(
                PromptProgressEvent(
                    prefill_tokens_processed=prefill_tokens_processed,
                    is_final=is_final,
                )
            )

        for response in generation_responses:
            result = state.active.get(response.uid)
            if result is None:
                continue

            result.rqueue.put(self._emit_response(result, response))
            if response.finish_reason is not None:
                keep_hot_cache = (
                    len(state.active) == 1 and not state.ready and not state.restoring
                )
                self._finish_response(result, response, keep_hot_cache)
                result.rqueue.put(None)
                del state.active[response.uid]

    def cancel_all_requests(self) -> None:
        state = self.state
        state.batch_generator.close()

        for result in state.active.values():
            result.rqueue.put(RequestCancelled("Model shutdown requested"))
        for request in state.pending:
            request.rqueue.put(RequestCancelled("Model shutdown requested"))
        for prepared_insert in state.ready:
            prepared_insert.request.rqueue.put(
                RequestCancelled("Model shutdown requested")
            )
        for request in state.restoring.values():
            request.rqueue.put(RequestCancelled("Model shutdown requested"))
