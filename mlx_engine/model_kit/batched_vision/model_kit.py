import copy
import gc
import logging
import sys
import time
import traceback
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Callable, Iterable

import mlx.core as mx
import mlx_lm
import mlx_vlm
from transformers.image_utils import ChannelDimension

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.model_kit.batched_vision.batch_generator import (
    BatchGenerator as LocalVlmBatchGenerator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    VlmPromptCacheCoordinator,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
    first_unsaved_prefix_cache_chunk_index,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    build_cached_prompt_kwargs,
    build_prompt_kwargs,
    prepare_prompt_inputs,
)
from mlx_engine.model_kit.batched_vision.cache_io_thread import PromptCacheIOThread
from mlx_engine.model_kit.batched_vision.request_lifecycle import (
    ActiveRequest,
    FailedRestore,
    GenerationThreadController,
    GenerationThreadState,
    GenerationRequest,
    PreparedInsert,
)
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.token import Token
from mlx_engine.utils.prompt_progress_events import (
    PromptProgressBeginEvent,
    PromptProgressEvent,
)
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_engine.vision_model_kit._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)

logger = logging.getLogger(__name__)
DEFAULT_MAX_SEQ_NUMS = 4


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
        max_seq_nums: int | None = DEFAULT_MAX_SEQ_NUMS,
        trust_remote_code: bool = False,
        seed: int | None = None,
    ):
        # External requests and internal generation events share one queue so
        # restore completions wake the generation thread without polling.
        self._requests = Queue()
        self._backend_exception = None
        self._generation_thread = None
        self._generation_thread_state: GenerationThreadState | None = None
        self._shutdown = Event()
        self._startup_complete = Event()
        self.prefill_step_size = prefill_step_size
        self._model_path = model_path
        if max_seq_nums is None:
            max_seq_nums = DEFAULT_MAX_SEQ_NUMS
        elif max_seq_nums < 1:
            max_seq_nums = 1
            logger.info(f"Setting concurrent request limit to {max_seq_nums}")
        self._max_seq_nums = max_seq_nums
        self._trust_remote_code = trust_remote_code
        self._seed = seed

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
        self._ensure_channel_first_if_fast_processor()

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
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer,
            model_path=self._model_path,
            model_type=self.model_type,
        )
        self.detokenizer = self.tokenizer.detokenizer

    def _new_detokenizer(self):
        """Reuse the expensive token map when the MLX detokenizer supports copying."""
        try:
            detokenizer = copy.copy(self.detokenizer)
        except Exception:
            detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        return detokenizer

    def _ensure_channel_first_if_fast_processor(self) -> None:
        if self.model_type != "lfm2-vl":
            return
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and getattr(image_processor, "is_fast", False):
            image_processor.input_data_format = ChannelDimension.FIRST

    def _load_model(self) -> None:
        self.model, _ = mlx_vlm.utils.load_model(
            self._model_path,
            lazy=False,
            trust_remote_code=self._trust_remote_code,
        )
        mx.synchronize()
        mx.clear_cache()

    def start(self):
        self._generation_thread = Thread(
            target=self._generate_with_exception_handling,
            name="mlx-engine-vlm-generation",
            daemon=True,
        )
        self._generation_thread.start()
        self._startup_complete.wait()
        if isinstance(self._backend_exception, Exception):
            raise self._backend_exception
        self._cache_io_thread.start()

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
        prompt_progress_reporter: PromptProgressReporter | None,
        top_logprobs: int,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: list,
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
                sampler=sampler,
                logits_processors=logits_processors,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
            )
        )

        def report_prompt_progress(response) -> bool:
            if prompt_progress_reporter is None:
                return True
            if isinstance(response, PromptProgressBeginEvent):
                return prompt_progress_reporter.begin(
                    is_draft=False,
                    cached_tokens=response.cached_tokens,
                    total_prompt_tokens=response.total_prompt_tokens,
                    prefill_tokens_processed=response.prefill_tokens_processed,
                )
            if response.is_final:
                return prompt_progress_reporter.finish(
                    is_draft=False,
                    prefill_tokens_processed=response.prefill_tokens_processed,
                )
            return prompt_progress_reporter.update(
                is_draft=False,
                prefill_tokens_processed=response.prefill_tokens_processed,
            )

        def _inner() -> Iterable[BatchedGenerationResponse]:
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(
                    response, (PromptProgressBeginEvent, PromptProgressEvent)
                ):
                    try:
                        should_continue = report_prompt_progress(response)
                    except Exception:
                        self.remove(request_id)
                        raise
                    if should_continue is False:
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
            if self._generation_thread_state is not None:
                self._generation_thread_state.batch_generator.close()
                self._generation_thread_state = None
            self._prompt_cache_coordinator.clear_hot_prompt_cache()
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.detokenizer = None
            self._generation_thread = None
            gc.collect()
            mx.clear_cache()

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
            self._generate()
        except Exception:
            err_string = f"Encountered fatal exception in the backend generation thread: {traceback.format_exc()}"
            logger.error(err_string)
            self._backend_exception = Exception(err_string)
            self._fail_all_requests(self._backend_exception)
            self._startup_complete.set()

            # Sleep to allow error messages to be logged and propagated to clients.
            time.sleep(3)
            sys.exit(1)

    def _make_batch_generator(self) -> LocalVlmBatchGenerator:
        vlm_tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        vlm_tokenizer.stopping_criteria.reset(list(self.tokenizer.eos_token_ids))
        return LocalVlmBatchGenerator(
            getattr(self.model, "language_model", self.model),
            vlm_tokenizer.stopping_criteria,
            max_tokens=10000000,
            # LM Studio owns the concurrency limit; do not use mlx-vlm's
            # internal batcher default here.
            completion_batch_size=self._max_seq_nums,
            prefill_step_size=(
                None
                if getattr(self.model, "no_chunked_prefill", False)
                else self.prefill_step_size
            ),
        )

    def _prepare_request_for_insert(self, request: GenerationRequest) -> PreparedInsert:
        prepared_prompt = prepare_prompt_inputs(
            prompt_tokens=request.prompt_tokens,
            images_b64=request.images_b64,
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
        all_tokens = []
        rope_deltas = None
        cached_prefix_len = 0
        if restored is not None:
            prompt_input_ids = full_prompt_input_ids[restored.cached_prefix_len :]
            prompt_kwargs = build_cached_prompt_kwargs(
                self.model,
                prepared_prompt,
                restored.cached_prefix_len,
                restored.rope_deltas,
            )
            inputs_embeds = prompt_kwargs.pop("inputs_embeds")
            cached_prefix_len = restored.cached_prefix_len
            cache = restored.prompt_cache
            all_tokens = full_prompt_input_ids[: restored.cached_prefix_len]
            rope_deltas = restored.rope_deltas
        else:
            prompt_kwargs = build_prompt_kwargs(self.model, prepared_prompt)
            inputs_embeds = prompt_kwargs.pop("inputs_embeds")

        if getattr(self.model, "no_chunked_prefill", False):
            # One-shot prefill skips exact intermediate prompt boundaries.
            prompt_progress_for_cache_chunks = prompt_token_count
        else:
            prompt_progress_for_cache_chunks = cached_prefix_len
        prefix_cache_chunks = build_prefix_cache_chunks(
            full_prompt_input_ids,
            prepared_prompt.image_spans,
        )
        next_prefix_cache_chunk_idx = first_unsaved_prefix_cache_chunk_index(
            prefix_cache_chunks,
            prompt_progress_for_cache_chunks,
        )
        prompt_cache_save_callback = (
            self._prompt_cache_coordinator.save_prompt_cache_snapshot
            if self._prompt_cache_store.can_store_records()
            else None
        )
        detokenizer = self._new_detokenizer()

        total_prompt_tokens = max(0, prompt_token_count - 1)
        cached_tokens = min(cached_prefix_len, total_prompt_tokens)
        request.rqueue.put(
            PromptProgressBeginEvent(
                cached_tokens=cached_tokens,
                total_prompt_tokens=total_prompt_tokens,
                prefill_tokens_processed=0,
            )
        )
        uid = batch_generator.insert(
            prompt_input_ids,
            inputs_embeds=inputs_embeds,
            max_tokens=request.max_tokens,
            top_logprobs=request.top_logprobs,
            sampler=request.sampler,
            logits_processors=request.logits_processors,
            cache=cache,
            all_tokens=all_tokens,
            rope_deltas=rope_deltas,
            prompt_kwargs=prompt_kwargs,
            prefix_cache_chunks=prefix_cache_chunks,
            next_prefix_cache_chunk_idx=next_prefix_cache_chunk_idx,
            image_spans=prepared_prompt.image_spans,
            prompt_cache_save_callback=prompt_cache_save_callback,
        )

        active[uid] = ActiveRequest(
            rqueue=request.rqueue,
            detokenizer=detokenizer,
            top_logprobs=request.top_logprobs,
            request_id=request.request_id,
            image_spans=prepared_prompt.image_spans,
            cached_tokens=cached_tokens,
        )

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

    def _finish_response(
        self, result: ActiveRequest, response, keep_hot_cache: bool
    ) -> None:
        if not keep_hot_cache:
            self._prompt_cache_coordinator.clear_hot_prompt_cache()

        if response.prompt_cache is None or response.all_tokens is None:
            return

        cache_store_budget_update = (
            self._prompt_cache_store.budget_update_from_completed_cache(
                response.prompt_cache
            )
        )
        self._cache_io_thread.enqueue_cache_store_budget_update(
            cache_store_budget_update
        )

        if keep_hot_cache:
            self._prompt_cache_coordinator.store_hot_prompt_cache(
                prompt_input_ids=response.all_tokens,
                image_spans=result.image_spans,
                prompt_cache=response.prompt_cache,
                rope_deltas=response.rope_deltas,
            )

    def _fail_all_requests(self, error: Exception) -> None:
        for item in GenerationThreadController.drain_queue(self._requests, None):
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
        set_seed(self._seed)

        if self.model is None:
            self._load_model()
        self._startup_complete.set()

        state = GenerationThreadState(batch_generator=self._make_batch_generator())
        self._generation_thread_state = state
        controller = GenerationThreadController(
            state=state,
            request_queue=self._requests,
            max_seq_nums=self._max_seq_nums,
            enqueue_restore=self._cache_io_thread.enqueue_restore,
            insert_prepared_request=self._insert_prepared_request,
            emit_response=self._emit_response,
            finish_response=self._finish_response,
        )

        while not self._shutdown.is_set():
            timeout = None if state.active or state.ready else 0.1
            controller.drain_generation_events(timeout)
            controller.insert_ready_requests()
            controller.admit_pending_requests()
            controller.step_generation()

        controller.cancel_all_requests()

    def __del__(self):
        self.shutdown()
