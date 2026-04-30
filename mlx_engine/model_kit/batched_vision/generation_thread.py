import logging
import traceback
from dataclasses import dataclass, field
from itertools import count
from queue import Empty as QueueEmpty
from queue import PriorityQueue
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable

import mlx.core as mx
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator as LocalVlmBatchGenerator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    RestoredPromptCache,
)
from mlx_engine.model_kit.batched_vision.prompt_inputs import PreparedPrompt
from mlx_engine.model_kit.batched_vision.prompt_cache.spill_cache import (
    VlmPromptSpillCache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PendingPromptCacheSave,
    PromptImageSpan,
)

logger = logging.getLogger(__name__)


_RESTORE_JOB_PRIORITY = 0
_SPILL_CAP_UPDATE_JOB_PRIORITY = 1
_SAVE_JOB_PRIORITY = 2
_SHUTDOWN_JOB_PRIORITY = -1


@dataclass
class GenerationRequest:
    rqueue: Queue
    prompt_tokens: list[int]
    request_id: str
    images_b64: list[str] | None
    max_image_size: tuple[int, int] | None
    sampler: Callable[[mx.array], mx.array]
    top_logprobs: int
    max_tokens: int


@dataclass
class PreparedInsert:
    request: GenerationRequest
    prepared_prompt: PreparedPrompt
    restored: RestoredPromptCache | None


@dataclass
class RestoreJob:
    request: GenerationRequest


@dataclass
class SaveJob:
    pending_save: PendingPromptCacheSave


@dataclass
class SpillCapUpdateJob:
    max_cache_bytes: int


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


@dataclass
class GenerationThreadState:
    batch_generator: LocalVlmBatchGenerator
    active: dict[int, ActiveRequest] = field(default_factory=dict)
    pending: list[GenerationRequest] = field(default_factory=list)
    ready: list[PreparedInsert] = field(default_factory=list)
    restoring: dict[str, GenerationRequest] = field(default_factory=dict)
    cancelled_restores: set[str] = field(default_factory=set)


class PromptCacheIOThread:
    """Runs background restore prep and blocking spill-cache commits."""

    def __init__(
        self,
        *,
        spill_cache: VlmPromptSpillCache,
        generation_queue: Queue,
        prepare_request: Callable[[GenerationRequest], PreparedInsert],
    ):
        self._spill_cache = spill_cache
        self._generation_queue = generation_queue
        self._prepare_request = prepare_request
        self._queue = PriorityQueue()
        self._sequence = count()
        self._thread = None
        self._closed = Event()

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._thread is None:
            self._close_spill_cache()
            return
        self._enqueue(_SHUTDOWN_JOB_PRIORITY, None)
        self._thread.join()

    def enqueue_restore(self, request: GenerationRequest) -> None:
        self._enqueue(_RESTORE_JOB_PRIORITY, RestoreJob(request))

    def enqueue_save(self, pending_save: PendingPromptCacheSave) -> None:
        # The generation thread already prepared arrays; this thread does disk I/O.
        self._enqueue(_SAVE_JOB_PRIORITY, SaveJob(pending_save))

    def enqueue_spill_cap_update(self, max_cache_bytes: int | None) -> None:
        if max_cache_bytes is None:
            return
        # Cap changes can evict blob-store records, so keep them on this thread.
        self._enqueue(
            _SPILL_CAP_UPDATE_JOB_PRIORITY,
            SpillCapUpdateJob(max_cache_bytes),
        )

    def _enqueue(self, priority: int, job: Any) -> None:
        self._queue.put((priority, next(self._sequence), job))

    def _close_spill_cache(self) -> None:
        self._spill_cache.close()

    def _discard_queued_jobs(self) -> None:
        while True:
            try:
                # Dropped spill jobs only hold immutable arrays or caps.
                self._queue.get_nowait()
            except QueueEmpty:
                return

    def _run(self) -> None:
        while True:
            _, _, job = self._queue.get()
            if job is None:
                self._discard_queued_jobs()
                self._close_spill_cache()
                return

            if isinstance(job, RestoreJob):
                try:
                    prepared_insert = self._prepare_request(job.request)
                except Exception as exc:
                    self._generation_queue.put(FailedRestore(job.request, exc))
                    continue

                self._generation_queue.put(prepared_insert)
                continue

            if isinstance(job, SaveJob):
                try:
                    self._spill_cache.commit_pending_save(job.pending_save)
                except Exception:
                    logger.error(
                        "Failed to commit pending prompt cache save:\n%s",
                        traceback.format_exc(),
                    )
                continue

            if isinstance(job, SpillCapUpdateJob):
                try:
                    self._spill_cache.commit_spill_cap_update(job.max_cache_bytes)
                except Exception:
                    logger.error(
                        "Failed to commit prompt spill cache cap:\n%s",
                        traceback.format_exc(),
                    )
