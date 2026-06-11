import logging
import traceback
from dataclasses import dataclass
from itertools import count
from queue import Empty as QueueEmpty
from queue import PriorityQueue
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable

from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PendingPromptCacheSave,
)
from mlx_engine.model_kit.batched_vision.request_lifecycle import (
    FailedRestore,
    GenerationRequest,
    PreparedInsert,
)
from mlx_engine.utils.mlx_threading import (
    install_mlx_compile_cache_cleanup_for_thread,
)

logger = logging.getLogger(__name__)


_RESTORE_JOB_PRIORITY = 0
_CACHE_STORE_BUDGET_UPDATE_JOB_PRIORITY = 1
_SAVE_JOB_PRIORITY = 2
_SHUTDOWN_JOB_PRIORITY = -1


@dataclass
class RestoreJob:
    request: GenerationRequest


@dataclass
class SaveJob:
    pending_save: PendingPromptCacheSave


@dataclass
class CacheStoreBudgetUpdateJob:
    max_cache_store_bytes: int


class PromptCacheIOThread:
    """Runs background restore prep and blocking cache store commits."""

    def __init__(
        self,
        *,
        cache_store: VlmPromptCacheStore,
        generation_queue: Queue,
        prepare_request: Callable[[GenerationRequest], PreparedInsert],
    ):
        self._cache_store = cache_store
        self._generation_queue = generation_queue
        self._prepare_request = prepare_request
        self._queue = PriorityQueue()
        self._sequence = count()
        self._thread = None
        self._closed = Event()

    def start(self) -> None:
        self._thread = Thread(
            target=self._run,
            name="mlx-engine-vlm-cache-io",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._thread is None:
            self._close_cache_store()
            return
        self._enqueue(_SHUTDOWN_JOB_PRIORITY, None)
        self._thread.join()
        self._close_cache_store()

    def enqueue_restore(self, request: GenerationRequest) -> None:
        self._enqueue(_RESTORE_JOB_PRIORITY, RestoreJob(request))

    def enqueue_save(self, pending_save: PendingPromptCacheSave) -> None:
        # The generation thread already prepared arrays; this thread does disk I/O.
        self._enqueue(_SAVE_JOB_PRIORITY, SaveJob(pending_save))

    def enqueue_cache_store_budget_update(
        self, max_cache_store_bytes: int | None
    ) -> None:
        if max_cache_store_bytes is None:
            return
        # Budget changes can evict blob-store records, so keep them on this thread.
        self._enqueue(
            _CACHE_STORE_BUDGET_UPDATE_JOB_PRIORITY,
            CacheStoreBudgetUpdateJob(max_cache_store_bytes),
        )

    def _enqueue(self, priority: int, job: Any) -> None:
        self._queue.put((priority, next(self._sequence), job))

    def _close_cache_store(self) -> None:
        self._cache_store.close()

    def _discard_queued_jobs(self) -> None:
        while True:
            try:
                # Dropped cache store jobs only hold immutable arrays or budgets.
                self._queue.get_nowait()
            except QueueEmpty:
                return

    def _run(self) -> None:
        install_mlx_compile_cache_cleanup_for_thread()
        while True:
            _, _, job = self._queue.get()
            if job is None:
                self._discard_queued_jobs()
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
                    self._cache_store.commit_pending_save(job.pending_save)
                except Exception:
                    logger.error(
                        "Failed to commit pending prompt cache save:\n%s",
                        traceback.format_exc(),
                    )
                continue

            if isinstance(job, CacheStoreBudgetUpdateJob):
                try:
                    self._cache_store.commit_budget_update(job.max_cache_store_bytes)
                except Exception:
                    logger.error(
                        "Failed to commit prompt cache store budget:\n%s",
                        traceback.format_exc(),
                    )
