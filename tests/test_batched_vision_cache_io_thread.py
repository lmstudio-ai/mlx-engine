import time
from queue import Queue

import mlx.core as mx

from mlx_engine.model_kit.batched_vision.cache_io_thread import PromptCacheIOThread
from mlx_engine.model_kit.batched_vision.request_lifecycle import (
    FailedRestore,
    GenerationRequest,
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


def _wait_until(condition) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


class _FakeCacheStore:
    def __init__(self, *, fail_save: bool = False):
        self.fail_save = fail_save
        self.calls = []
        self.closed = False

    def commit_pending_save(self, pending_save):
        self.calls.append(("save", pending_save))
        if self.fail_save:
            raise RuntimeError("save failed")

    def commit_budget_update(self, max_cache_store_bytes: int):
        self.calls.append(("budget", max_cache_store_bytes))

    def close(self):
        self.closed = True


def test_cache_io_thread_commits_queued_save_before_restore():
    """Queued saves should be visible to later restores before budget work runs."""
    store = _FakeCacheStore()
    generation_queue = Queue()
    request = _request("restore")
    thread = PromptCacheIOThread(
        cache_store=store,
        generation_queue=generation_queue,
        prepare_request=_prepared,
    )

    thread.enqueue_save("save")
    thread.enqueue_cache_store_budget_update(123)
    thread.enqueue_restore(request)
    thread.start()

    try:
        result = generation_queue.get(timeout=1.0)
        _wait_until(lambda: len(store.calls) == 2)
    finally:
        thread.close()

    assert isinstance(result, PreparedInsert)
    assert result.request is request
    assert store.calls == [("save", "save"), ("budget", 123)]
    assert store.closed


def test_cache_io_thread_restore_error_posts_failed_restore_and_continues():
    """A bad restore reports failure but does not kill the cache I/O thread."""
    generation_queue = Queue()
    bad_request = _request("bad")
    good_request = _request("good")

    def prepare(request):
        if request is bad_request:
            raise RuntimeError("restore failed")
        return _prepared(request)

    thread = PromptCacheIOThread(
        cache_store=_FakeCacheStore(),
        generation_queue=generation_queue,
        prepare_request=prepare,
    )
    thread.enqueue_restore(bad_request)
    thread.enqueue_restore(good_request)
    thread.start()

    try:
        failed = generation_queue.get(timeout=1.0)
        prepared = generation_queue.get(timeout=1.0)
    finally:
        thread.close()

    assert isinstance(failed, FailedRestore)
    assert failed.request is bad_request
    assert str(failed.error) == "restore failed"
    assert isinstance(prepared, PreparedInsert)
    assert prepared.request is good_request


def test_cache_io_thread_save_error_does_not_kill_worker():
    """Save failures are best-effort; later restores should still complete."""
    store = _FakeCacheStore(fail_save=True)
    generation_queue = Queue()
    request = _request("restore")
    thread = PromptCacheIOThread(
        cache_store=store,
        generation_queue=generation_queue,
        prepare_request=_prepared,
    )
    thread.start()

    try:
        thread.enqueue_save("bad-save")
        _wait_until(lambda: store.calls == [("save", "bad-save")])
        thread.enqueue_restore(request)
        prepared = generation_queue.get(timeout=1.0)
    finally:
        thread.close()

    assert isinstance(prepared, PreparedInsert)
    assert prepared.request is request


def test_cache_io_thread_close_before_start_closes_store():
    """Closing an unstarted cache I/O thread still releases the blob store."""
    store = _FakeCacheStore()
    thread = PromptCacheIOThread(
        cache_store=store,
        generation_queue=Queue(),
        prepare_request=_prepared,
    )

    thread.close()

    assert store.closed
