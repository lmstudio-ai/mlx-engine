from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
import threading

import mlx_engine.model_kit.batched_model_kit as batched_model_kit_module
from mlx_engine.model_kit.batched_model_kit import BatchedModelKit
from mlx_engine.model_kit.batched_model_kit_types import GenerationRequest


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        rqueue=Queue(),
        prompt_tokens=[1, 2, 3],
        request_id=request_id,
        samplers=None,
        logits_processors=[],
        top_logprobs=0,
        max_tokens=10,
    )


def test_pending_request_is_admitted_before_the_next_generation_step(monkeypatch):
    first_request = _request("first")
    second_request = _request("second")
    inserted_request_count = 0
    generation_step_count = 0

    class FakeBatchGenerator:
        stream = object()

        def __init__(self, *_args, **_kwargs):
            pass

        def insert(self, *_args, **_kwargs):
            nonlocal inserted_request_count
            uid = inserted_request_count
            inserted_request_count += 1
            if inserted_request_count == 2:
                model_kit._shutdown.set()
            return (uid,)

        def next(self):
            nonlocal generation_step_count
            generation_step_count += 1
            if generation_step_count == 1:
                model_kit._requests.put(second_request)
                return [SimpleNamespace(uid=0, progress=(1, 3))], []
            model_kit._shutdown.set()
            return [], []

    monkeypatch.setattr(batched_model_kit_module, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        batched_model_kit_module,
        "_prepare_prompt_cache_for_generation",
        lambda *_args: (None, [], [1, 2, 3]),
    )
    monkeypatch.setattr(
        batched_model_kit_module,
        "install_mlx_compile_cache_cleanup_for_thread",
        lambda: None,
    )
    monkeypatch.setattr(batched_model_kit_module, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        batched_model_kit_module.mx,
        "stream",
        lambda _stream: nullcontext(),
    )

    model_kit = BatchedModelKit.__new__(BatchedModelKit)
    model_kit.model = object()
    model_kit.tokenizer = SimpleNamespace(detokenizer=object(), eos_token_ids=[])
    model_kit._requests = Queue()
    model_kit._requests.put(first_request)
    model_kit._prompt_cache = object()
    model_kit._batch_results = {}
    model_kit._backend_exception = None
    model_kit._generation_thread = None
    model_kit._shutdown = threading.Event()
    model_kit._startup_complete = threading.Event()
    model_kit._seed = None
    model_kit._max_seq_nums = 4
    model_kit._prefill_step_size = 512
    model_kit._max_kv_size = 2048

    scheduler_thread = threading.Thread(target=model_kit._generate)
    scheduler_thread.start()
    scheduler_thread.join(timeout=2)

    assert not scheduler_thread.is_alive()
    assert inserted_request_count == 2
    assert generation_step_count == 1
