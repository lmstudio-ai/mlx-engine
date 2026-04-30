import logging
import sys
import time
import traceback
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Thread
from typing import Callable, Iterable

import mlx.core as mx
import mlx_lm
import mlx_vlm

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator as LocalVlmBatchGenerator,
    use_generation_stream,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    VlmPromptCacheCoordinator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    build_cached_prompt_kwargs,
    build_prompt_kwargs,
    prepare_prompt_inputs,
)
from mlx_engine.model_kit.batched_vision.generation_thread import (
    ActiveRequest,
    FailedRestore,
    GenerationThreadState,
    GenerationRequest,
    PreparedInsert,
    PromptCacheIOThread,
)
from mlx_engine.utils.generation_helpers import MAX_TOP_LOGPROBS
from mlx_engine.utils.token import Token
from mlx_engine.vision_model_kit._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)

logger = logging.getLogger(__name__)


class BatchedVisionModelKit:
    """
    VLM batching backend built on a local mlx-vlm-style batcher.

    A single worker thread owns the local VLM batcher and handles
    both text-only and basic image requests.
    """

    model = None
    processor = None
    tokenizer = None
    detokenizer = None
    model_type: str | None = None

    def __init__(
        self,
        model_path: Path,
        prefill_step_size: int,
        max_kv_size: int | None = None,
        max_seq_nums: int = 4,
        trust_remote_code: bool = False,
    ):
        # External requests and internal generation events share one queue so
        # restore completions wake the generation thread without polling.
        self._requests = Queue()
        self._backend_exception = None
        self._generation_thread = None
        self._generation_thread_state: GenerationThreadState | None = None
        self._shutdown = Event()
        self.prefill_step_size = prefill_step_size
        self._model_path = model_path
        self._max_seq_nums = max_seq_nums
        self._trust_remote_code = trust_remote_code

        fix_qwen2_5_vl_image_processor(model_path)
        fix_qwen2_vl_preprocessor(model_path)

        self.config = mlx_vlm.utils.load_config(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model_type = self.config.get("model_type")
        self._prompt_cache_store = VlmPromptCacheStore(max_kv_size=max_kv_size)
        self._cache_io_thread = PromptCacheIOThread(
            cache_store=self._prompt_cache_store,
            generation_queue=self._requests,
            prepare_request=self._prepare_request_for_insert,
        )
        self._prompt_cache_coordinator = VlmPromptCacheCoordinator(
            self._prompt_cache_store,
            self._cache_io_thread.enqueue_save,
        )

        # Keep tokenizer/config (and the pure-Python processor) on the caller
        # thread, but load the MLX-backed model on the generation thread. MLX
        # 0.31.2 tolerates the threaded stream usage we want, but async_eval
        # still trips if the underlying MLX arrays/modules were created on a
        # different thread.
        self._init_tokenizer_only()
        self.processor = mlx_vlm.utils.load_processor(
            self._model_path,
            True,
            eos_token_ids=self._get_eos_token_ids(),
            trust_remote_code=self._trust_remote_code,
        )
        image_processor = mlx_vlm.utils.load_image_processor(
            self._model_path,
            trust_remote_code=self._trust_remote_code,
        )
        if image_processor is not None:
            self.processor.image_processor = image_processor

    def _get_eos_token_ids(self) -> list[int] | None:
        eos_token_ids_raw = self.config.get("eos_token_id")
        if eos_token_ids_raw is None:
            eos_token_ids_raw = self.config.get("text_config", {}).get("eos_token_id")
        if eos_token_ids_raw is None:
            return None
        if isinstance(eos_token_ids_raw, int):
            return [eos_token_ids_raw]
        return list(dict.fromkeys(eos_token_ids_raw))

    def _init_tokenizer_only(self) -> None:
        self.tokenizer = mlx_lm.tokenizer_utils.load(
            self._model_path, eos_token_ids=self._get_eos_token_ids()
        )
        self.detokenizer = self.tokenizer.detokenizer

    def _load_model(self) -> None:
        self.model, _ = mlx_vlm.utils.load_model(
            self._model_path,
            lazy=False,
            trust_remote_code=self._trust_remote_code,
        )
        mx.clear_cache()

    def start(self):
        mx.synchronize()
        self._cache_io_thread.start()
        self._generation_thread = Thread(
            target=self._generate_with_exception_handling, daemon=True
        )
        self._generation_thread.start()

    def tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def generate(
        self,
        *,
        prompt_tokens: list[int],
        request_id: str,
        images_b64: list[str] | None,
        max_image_size: tuple[int, int] | None,
        prompt_progress_callback,
        top_logprobs: int,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
    ):
        if self._shutdown.is_set():
            raise RuntimeError("Cannot accept new requests when model is shutdown")
        if isinstance(self._backend_exception, Exception):
            raise self._backend_exception

        response_queue = Queue()
        self._requests.put(
            GenerationRequest(
                rqueue=response_queue,
                prompt_tokens=prompt_tokens,
                request_id=request_id,
                images_b64=images_b64,
                max_image_size=max_image_size,
                sampler=sampler,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
            )
        )

        def _inner() -> Iterable[BatchedGenerationResponse]:
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if prompt_progress_callback is not None:
                        if prompt_progress_callback(*response) is False:
                            # Prompt cancellation is cooperative at chunk boundaries.
                            self.remove(request_id)
                            raise RequestCancelled()
                    continue
                yield response

        return _inner()

    def remove(self, request_id: str):
        self._requests.put(CancelGenerationRequest(request_id))

    def shutdown(self):
        if not self._shutdown.is_set():
            self._shutdown.set()
            if self._generation_thread:
                self._generation_thread.join()
            self._cache_io_thread.close()

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        return False

    def load_draft_model(self, path: str | Path) -> None:
        raise ValueError(
            "Speculative decoding is not currently supported for batched vision models"
        )

    def unload_draft_model(self) -> None:
        raise ValueError(
            "Speculative decoding is not currently supported for batched vision models"
        )

    def _generate_with_exception_handling(self):
        try:
            if self.model is None:
                self._load_model()
            self._generate()
        except Exception:
            err_string = f"Encountered fatal exception in the backend generation thread: {traceback.format_exc()}"
            logger.error(err_string)
            self._backend_exception = Exception(err_string)
            self._fail_all_requests(self._backend_exception)

            # Sleep to allow error messages to be logged and propagated to clients.
            time.sleep(3)
            sys.exit(1)

    def _make_batch_generator(self) -> LocalVlmBatchGenerator:
        vlm_tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        vlm_tokenizer.stopping_criteria.add_eos_token_ids(
            list(self.tokenizer.eos_token_ids)
        )
        return LocalVlmBatchGenerator(
            getattr(self.model, "language_model", self.model),
            vlm_tokenizer.stopping_criteria,
            max_tokens=10000000,
            completion_batch_size=self._max_seq_nums,
            prefill_step_size=(
                None
                if getattr(self.model, "no_chunked_prefill", False)
                else self.prefill_step_size
            ),
            top_logprobs_k=MAX_TOP_LOGPROBS,
        )

    def _prepare_request_for_insert(self, request: GenerationRequest) -> PreparedInsert:
        prepared_prompt = prepare_prompt_inputs(
            prompt_tokens=request.prompt_tokens,
            images_b64=request.images_b64,
            max_image_size=request.max_image_size,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config,
        )
        restored = self._prompt_cache_coordinator.restore(
            prompt_input_ids=prepared_prompt.prompt_input_ids,
            image_spans=prepared_prompt.image_spans,
        )
        return PreparedInsert(
            request=request,
            prepared_prompt=prepared_prompt,
            restored=restored,
        )

    def _insert_prepared_request(
        self,
        batch_generator: LocalVlmBatchGenerator,
        prepared_insert: PreparedInsert,
        active: dict[int, ActiveRequest],
    ) -> None:
        request = prepared_insert.request
        prepared_prompt = prepared_insert.prepared_prompt
        restored = prepared_insert.restored
        full_prompt_input_ids = prepared_prompt.prompt_input_ids
        prompt_token_count = len(full_prompt_input_ids)

        prompt_input_ids = full_prompt_input_ids
        prompt_kwargs = None
        inputs_embeds = None
        cache = None
        all_tokens = None
        rope_deltas = None
        prompt_progress = 0
        with use_generation_stream():
            if restored is not None:
                prompt_input_ids = full_prompt_input_ids[restored.cached_prefix_len :]
                prompt_kwargs = build_cached_prompt_kwargs(
                    self.model,
                    prepared_prompt,
                    restored.cached_prefix_len,
                    restored.rope_deltas,
                )
                inputs_embeds = prompt_kwargs.pop("inputs_embeds")
                prompt_progress = restored.cached_prefix_len
                cache = restored.prompt_cache
                all_tokens = full_prompt_input_ids[: restored.cached_prefix_len]
                rope_deltas = restored.rope_deltas
            else:
                prompt_kwargs = build_prompt_kwargs(self.model, prepared_prompt)
                inputs_embeds = prompt_kwargs.pop("inputs_embeds")

        cache_save_points = self._prompt_cache_coordinator.save_points_after(
            prompt_input_ids=full_prompt_input_ids,
            image_spans=prepared_prompt.image_spans,
            prompt_progress=prompt_progress,
        )
        prompt_cache_save_callback = None
        if cache_save_points:
            prompt_cache_save_callback = (
                self._prompt_cache_coordinator.make_save_callback(
                    image_spans=prepared_prompt.image_spans
                )
            )

        request.rqueue.put((prompt_progress, prompt_token_count))
        uid = batch_generator.insert(
            prompt_input_ids,
            inputs_embeds=inputs_embeds,
            max_tokens=request.max_tokens,
            sampler=request.sampler,
            cache=cache,
            all_tokens=all_tokens,
            rope_deltas=rope_deltas,
            prompt_kwargs=prompt_kwargs,
            cache_save_points=cache_save_points,
            prompt_cache_save_callback=prompt_cache_save_callback,
        )

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        active[uid] = ActiveRequest(
            rqueue=request.rqueue,
            detokenizer=detokenizer,
            top_logprobs=request.top_logprobs,
            request_id=request.request_id,
            image_spans=prepared_prompt.image_spans,
        )

    def _cancel_request(
        self,
        request_id: str,
        state: GenerationThreadState,
    ) -> bool:
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

    def _emit_response(
        self, result: ActiveRequest, response
    ) -> BatchedGenerationResponse:
        detokenizer = result.detokenizer
        if response.finish_reason != "stop":
            detokenizer.add_token(response.token)
        if response.finish_reason is not None:
            detokenizer.finalize()

        top_logprobs = None
        if result.top_logprobs > 0 and response.top_logprobs is not None:
            top_logprobs = [
                Token(
                    id=int(token_id),
                    text=self.tokenizer.decode(token_id),
                    logprob=float(logprob),
                )
                for token_id, logprob in response.top_logprobs[: result.top_logprobs]
            ]

        return BatchedGenerationResponse(
            text=detokenizer.last_segment,
            token=int(response.token),
            token_logprob=float(response.token_logprob),
            top_logprobs=top_logprobs,
            finish_reason=response.finish_reason,
            from_draft=False,
        )

    @staticmethod
    def _drain_queue(queue: Queue, timeout: float | None) -> list:
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

    def _handle_prepared_event(
        self,
        item: PreparedInsert | FailedRestore,
        state: GenerationThreadState,
    ) -> None:
        request_id = item.request.request_id
        state.restoring.pop(request_id, None)

        if request_id in state.cancelled_restores:
            state.cancelled_restores.discard(request_id)
            return

        if isinstance(item, FailedRestore):
            item.request.rqueue.put(item.error)
            return

        state.ready.append(item)

    @staticmethod
    def _reserved_slots(state: GenerationThreadState) -> int:
        return len(state.active) + len(state.ready) + len(state.restoring)

    def _drain_generation_events(
        self, state: GenerationThreadState, timeout: float | None
    ):
        for item in self._drain_queue(self._requests, timeout):
            if isinstance(item, CancelGenerationRequest):
                if not self._cancel_request(item.request_id, state):
                    logger.warning(f"Could not cancel request_id={item.request_id}")
                continue

            if isinstance(item, (PreparedInsert, FailedRestore)):
                self._handle_prepared_event(item, state)
                continue

            state.pending.append(item)

    def _insert_ready_requests(self, state: GenerationThreadState) -> None:
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

    def _admit_pending_requests(self, state: GenerationThreadState) -> None:
        if self._reserved_slots(state) >= self._max_seq_nums:
            return

        next_pending = []
        for request in state.pending:
            if self._reserved_slots(state) < self._max_seq_nums:
                state.restoring[request.request_id] = request
                self._cache_io_thread.enqueue_restore(request)
            else:
                next_pending.append(request)
        state.pending = next_pending

    def _step_generation(self, state: GenerationThreadState) -> None:
        if not state.active:
            return

        prompt_responses, generation_responses = state.batch_generator.next()
        for response in prompt_responses:
            result = state.active.get(response.uid)
            if result is None:
                continue
            processed, total = response.progress
            result.rqueue.put((min(processed, total), total))

        for response in generation_responses:
            result = state.active.get(response.uid)
            if result is None:
                continue

            result.rqueue.put(self._emit_response(result, response))
            if response.finish_reason is not None:
                if (
                    response.prompt_cache is not None
                    and response.all_tokens is not None
                ):
                    cache_store_budget_update = (
                        self._prompt_cache_store.budget_update_from_completed_cache(
                            response.prompt_cache
                        )
                    )
                    self._cache_io_thread.enqueue_cache_store_budget_update(
                        cache_store_budget_update
                    )
                    self._prompt_cache_coordinator.store_hot_prompt_cache(
                        prompt_input_ids=response.all_tokens,
                        image_spans=result.image_spans,
                        prompt_cache=response.prompt_cache,
                        rope_deltas=(
                            None
                            if response.rope_deltas is None
                            else response.rope_deltas
                        ),
                    )
                result.rqueue.put(None)
                del state.active[response.uid]

    def _cancel_all_requests(self, state: GenerationThreadState) -> None:
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

    def _fail_all_requests(self, error: Exception) -> None:
        for item in self._drain_queue(self._requests, None):
            if isinstance(item, GenerationRequest):
                item.rqueue.put(error)
            elif isinstance(item, (PreparedInsert, FailedRestore)):
                item.request.rqueue.put(error)

        state = self._generation_thread_state
        if state is None:
            return

        for result in state.active.values():
            result.rqueue.put(error)
        for request in state.pending:
            request.rqueue.put(error)
        for prepared_insert in state.ready:
            prepared_insert.request.rqueue.put(error)
        for request in state.restoring.values():
            request.rqueue.put(error)

    def _generate(self):
        state = GenerationThreadState(batch_generator=self._make_batch_generator())
        self._generation_thread_state = state

        while not self._shutdown.is_set():
            timeout = None if state.active or state.ready else 0.1
            self._drain_generation_events(state, timeout)
            self._insert_ready_requests(state)
            self._admit_pending_requests(state)
            self._step_generation(state)

        self._cancel_all_requests(state)

    def __del__(self):
        self.shutdown()
