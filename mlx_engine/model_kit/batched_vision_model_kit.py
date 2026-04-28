import hashlib
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import PriorityQueue
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Iterable, Optional

import mlx.core as mx
import mlx_lm
import mlx_vlm

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.model_kit.vlm_batch_generator import (
    BatchGenerator as LocalVlmBatchGenerator,
)
from mlx_engine.model_kit.vlm_prompt_cache_coordinator import (
    RestoredPromptCache,
    VlmPromptCacheCoordinator,
)
from mlx_engine.model_kit.vlm_prompt_cache_types import (
    PendingPromptCacheSave,
    PromptImageSpan,
)
from mlx_engine.model_kit.vlm_prompt_spill_cache import (
    VlmPromptSpillCache,
)
from mlx_engine.utils.generation_helpers import MAX_TOP_LOGPROBS, create_sampler
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize
from mlx_engine.utils.token import Token
from mlx_engine.vision_model_kit._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)

logger = logging.getLogger(__name__)


MLX_VLM_BATCHED_VISION_ENV_VAR = "MLX_ENGINE_USE_MLX_VLM_BATCHED_VISION"
MLX_VLM_BATCHED_VISION_DISK_CACHE_ENV_VAR = (
    "MLX_ENGINE_USE_MLX_VLM_BATCHED_VISION_DISK_CACHE"
)
_RESTORE_JOB_PRIORITY = 0
_SAVE_JOB_PRIORITY = 1
_SHUTDOWN_JOB_PRIORITY = -1


def is_mlx_vlm_batched_vision_enabled() -> bool:
    raw = os.environ.get(MLX_VLM_BATCHED_VISION_ENV_VAR, "")
    return raw.lower() in {"1", "true", "yes", "on"}


def is_mlx_vlm_batched_vision_disk_cache_enabled() -> bool:
    raw = os.environ.get(MLX_VLM_BATCHED_VISION_DISK_CACHE_ENV_VAR, "")
    return (
        raw.lower() in {"1", "true", "yes", "on"}
        and is_mlx_vlm_batched_vision_enabled()
    )


@dataclass
class _GenerationRequest:
    rqueue: Queue
    prompt_tokens: list[int]
    request_id: str
    images_b64: list[str] | None
    max_image_size: tuple[int, int] | None
    temp: float
    top_p: float
    top_k: int
    min_p: float
    min_tokens_to_keep: int
    top_logprobs: int
    max_tokens: int

    @property
    def sampling_key(self) -> tuple[float, float, int, float, int]:
        return (
            self.temp,
            self.top_p,
            self.top_k,
            self.min_p,
            self.min_tokens_to_keep,
        )


@dataclass
class _PreparedPrompt:
    prompt_input_ids: list[int]
    raw_inputs: Optional[dict[str, Any]]
    image_spans: list[PromptImageSpan]


@dataclass
class _PreparedInsert:
    request: _GenerationRequest
    prepared_prompt: _PreparedPrompt
    restored: RestoredPromptCache | None


@dataclass
class _RestoreJob:
    request: _GenerationRequest


@dataclass
class _SaveJob:
    pending_save: PendingPromptCacheSave


@dataclass
class _FailedRestore:
    request: _GenerationRequest
    error: Exception


@dataclass
class _SchedulerState:
    batch_generator: Optional[LocalVlmBatchGenerator] = None
    active: dict[int, dict] = field(default_factory=dict)
    pending: list[_GenerationRequest] = field(default_factory=list)
    ready: list[_PreparedInsert] = field(default_factory=list)
    restoring: dict[str, _GenerationRequest] = field(default_factory=dict)
    cancelled_restores: set[str] = field(default_factory=set)
    current_sampling_key: Any = None


class _PromptCacheActor:
    """Runs background restore prep and blocking spill-cache commits."""

    def __init__(
        self,
        *,
        spill_cache: VlmPromptSpillCache | None,
        scheduler_queue: Queue,
        prepare_request: Callable[[_GenerationRequest], _PreparedInsert],
    ):
        self._spill_cache = spill_cache
        self._scheduler_queue = scheduler_queue
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

    def enqueue_restore(self, request: _GenerationRequest) -> None:
        self._enqueue(_RESTORE_JOB_PRIORITY, _RestoreJob(request))

    def enqueue_save(self, pending_save: PendingPromptCacheSave) -> None:
        if self._spill_cache is None:
            return
        # The scheduler already prepared arrays; the actor handles disk I/O.
        self._enqueue(_SAVE_JOB_PRIORITY, _SaveJob(pending_save))

    def _enqueue(self, priority: int, job: Any) -> None:
        self._queue.put((priority, next(self._sequence), job))

    def _close_spill_cache(self) -> None:
        if self._spill_cache is not None:
            self._spill_cache.close()

    def _discard_queued_jobs(self) -> None:
        while True:
            try:
                # Dropped save jobs only hold immutable arrays; draining the
                # queue releases them without touching actor-owned cache state.
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

            if isinstance(job, _RestoreJob):
                try:
                    prepared_insert = self._prepare_request(job.request)
                except Exception as exc:
                    self._scheduler_queue.put(_FailedRestore(job.request, exc))
                    continue

                self._scheduler_queue.put(prepared_insert)
                continue

            if isinstance(job, _SaveJob):
                try:
                    self._spill_cache.commit_pending_save(job.pending_save)
                except Exception:
                    logger.error(
                        "Failed to commit pending prompt cache save:\n%s",
                        traceback.format_exc(),
                    )


class BatchedVisionModelKit:
    """
    Feature-flagged VLM batching backend built on a local mlx-vlm-style batcher.

    V1 deliberately keeps the scope narrow: a single worker thread owns the
    local VLM batcher and handles both text-only and basic image requests.
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
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        max_seq_nums: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
        # External requests and internal scheduler events share one queue so
        # restore completions wake the scheduler without a second polling path.
        self._requests = Queue()
        self._backend_exception = None
        self._generation_thread = None
        self._shutdown = Event()
        self.prefill_step_size = prefill_step_size
        self.vocab_only = vocab_only
        self._model_path = model_path
        self._max_seq_nums = max_seq_nums if max_seq_nums and max_seq_nums > 0 else 1
        self._trust_remote_code = trust_remote_code

        fix_qwen2_5_vl_image_processor(model_path)
        fix_qwen2_vl_preprocessor(model_path)

        self.config = mlx_vlm.utils.load_config(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model_type = self.config.get("model_type")
        self._prompt_spill_cache = (
            None
            if vocab_only or not is_mlx_vlm_batched_vision_disk_cache_enabled()
            else VlmPromptSpillCache(max_kv_size=max_kv_size)
        )
        self._prompt_cache_coordinator = (
            None
            if self._prompt_spill_cache is None
            else VlmPromptCacheCoordinator(
                self._prompt_spill_cache,
                self._enqueue_pending_spill_save,
            )
        )
        self._background_actor = (
            None
            if vocab_only
            else _PromptCacheActor(
                spill_cache=self._prompt_spill_cache,
                scheduler_queue=self._requests,
                prepare_request=self._prepare_request_for_insert,
            )
        )

        # Keep tokenizer/config (and the pure-Python processor) on the caller
        # thread, but load the MLX-backed model on the scheduler thread. MLX
        # 0.31.2 tolerates the threaded stream usage we want, but async_eval
        # still trips if the underlying MLX arrays/modules were created on a
        # different thread.
        self._init_tokenizer_only()
        if not vocab_only:
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
        return list(set(eos_token_ids_raw))

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
        if self.vocab_only:
            return
        mx.synchronize()
        self._background_actor.start()
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
        temp: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        min_tokens_to_keep: Optional[int],
    ):
        if self.vocab_only:
            raise RuntimeError("Cannot generate from a vocab-only model")
        if self._shutdown.is_set():
            raise RuntimeError("Cannot accept new requests when model is shutdown")
        if isinstance(self._backend_exception, Exception):
            raise self._backend_exception

        response_queue = Queue()
        self._requests.put(
            _GenerationRequest(
                rqueue=response_queue,
                prompt_tokens=prompt_tokens,
                request_id=request_id,
                images_b64=images_b64,
                max_image_size=max_image_size,
                temp=0.0 if temp is None else temp,
                top_p=0.0 if top_p is None else top_p,
                top_k=0 if top_k is None else top_k,
                min_p=0.0 if min_p is None else min_p,
                min_tokens_to_keep=(
                    1 if min_tokens_to_keep is None else min_tokens_to_keep
                ),
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
                        prompt_progress_callback(*response)
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
            if self._background_actor is not None:
                self._background_actor.close()

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def is_cross_prompt_cache_active(self) -> bool:
        return False

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
            err_string = f"Encountered fatal exception in the backend scheduler: {traceback.format_exc()}"
            logger.error(err_string)
            self._backend_exception = Exception(err_string)

            # Sleep to allow error messages to be logged and propagated to clients.
            time.sleep(3)
            sys.exit(1)

    def _make_batch_generator(
        self, request: _GenerationRequest
    ) -> LocalVlmBatchGenerator:
        sampler = create_sampler(
            request.temp,
            request.top_p,
            request.min_p,
            request.min_tokens_to_keep,
            request.top_k,
        )
        return LocalVlmBatchGenerator(
            getattr(self.model, "language_model", self.model),
            self.processor,
            max_tokens=10000000,
            stop_tokens=list(self.tokenizer.eos_token_ids),
            sampler=sampler,
            completion_batch_size=self._max_seq_nums,
            prefill_batch_size=1,
            prefill_step_size=(
                None
                if getattr(self.model, "no_chunked_prefill", False)
                else self.prefill_step_size
            ),
            top_logprobs_k=MAX_TOP_LOGPROBS,
        )

    def _get_image_token_index(self) -> int | None:
        for value in (
            self.config.get("image_token_index"),
            self.config.get("image_token_id"),
            self.config.get("vision_config", {}).get("image_token_id"),
        ):
            if value is not None:
                return value
        return None

    def _hash_resized_image(self, image) -> str:
        digest = hashlib.sha256()
        digest.update(image.mode.encode())
        digest.update(f"{image.size[0]}x{image.size[1]}".encode())
        digest.update(image.tobytes())
        return digest.hexdigest()

    def _get_image_spans(
        self, prompt_input_ids: list[int], image_hashes: list[str]
    ) -> list[PromptImageSpan]:
        if not image_hashes:
            return []

        image_token_index = self._get_image_token_index()
        if image_token_index is None:
            # Some processors do not expose a stable image sentinel; keep the
            # cache correct by making the whole prompt image-dependent.
            return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

        token_spans = []
        span_start = None
        for i, token_id in enumerate(prompt_input_ids):
            if token_id == image_token_index:
                if span_start is None:
                    span_start = i
            elif span_start is not None:
                token_spans.append((span_start, i))
                span_start = None
        if span_start is not None:
            token_spans.append((span_start, len(prompt_input_ids)))

        if len(token_spans) != len(image_hashes):
            # Mismatched processor output is rare, but wrong image reuse is worse
            # than missing a cache hit.
            return [PromptImageSpan(0, len(prompt_input_ids), "|".join(image_hashes))]

        return [
            PromptImageSpan(start, end, image_hash)
            for (start, end), image_hash in zip(token_spans, image_hashes)
        ]

    def _prepare_prompt_inputs(self, request: _GenerationRequest) -> _PreparedPrompt:
        prompt_tokens = request.prompt_tokens
        if len(prompt_tokens) == 0:
            prompt_tokens = self.tokenize(" ")

        if not request.images_b64:
            return _PreparedPrompt(
                prompt_input_ids=list(prompt_tokens),
                raw_inputs=None,
                image_spans=[],
            )

        # Keep image decode and prompt prep on the worker thread for now.
        prompt = self.tokenizer.decode(prompt_tokens) or " "
        images = custom_resize(
            convert_to_pil(request.images_b64),
            max_size=request.max_image_size,
        )
        raw_inputs = mlx_vlm.prepare_inputs(
            processor=self.processor,
            images=images,
            prompts=prompt,
            image_token_index=self._get_image_token_index(),
            resize_shape=None,
        )
        prompt_input_ids = raw_inputs["input_ids"].squeeze(0).tolist()
        image_hashes = [self._hash_resized_image(image) for image in images]
        return _PreparedPrompt(
            prompt_input_ids=prompt_input_ids,
            raw_inputs=raw_inputs,
            image_spans=self._get_image_spans(
                prompt_input_ids,
                image_hashes,
            ),
        )

    def _build_prompt_kwargs(self, prepared_prompt: _PreparedPrompt) -> dict:
        if prepared_prompt.raw_inputs is None:
            input_ids = mx.array(prepared_prompt.prompt_input_ids, dtype=mx.int32)[
                None, :
            ]
            embedding_output = self.model.get_input_embeddings(input_ids)
            return embedding_output.to_dict()

        raw_inputs = prepared_prompt.raw_inputs
        input_ids = raw_inputs["input_ids"]
        pixel_values = raw_inputs.get("pixel_values")
        attention_mask = raw_inputs.get("attention_mask")
        data_kwargs = {
            key: value
            for key, value in raw_inputs.items()
            if key not in {"input_ids", "pixel_values", "attention_mask"}
        }
        embedding_output = self.model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=attention_mask,
            **data_kwargs,
        )
        return {
            **data_kwargs,
            **embedding_output.to_dict(),
        }

    def _build_cached_prompt_kwargs(
        self,
        prepared_prompt: _PreparedPrompt,
        cached_prefix_len: int,
        rope_deltas: Optional[Any],
    ) -> dict:
        prompt_input_ids = prepared_prompt.prompt_input_ids[cached_prefix_len:]
        if prepared_prompt.raw_inputs is not None:
            prompt_kwargs = self._build_prompt_kwargs(prepared_prompt)
            # Keep full-prompt model side state, but prefill only the suffix.
            prompt_kwargs["inputs_embeds"] = prompt_kwargs["inputs_embeds"][
                :, cached_prefix_len:
            ]
            return prompt_kwargs

        input_ids = mx.array(prompt_input_ids, dtype=mx.int32)[None, :]
        embedding_output = self.model.get_input_embeddings(input_ids)
        prompt_kwargs = embedding_output.to_dict()

        # Prefix restores carry the tiny RoPE delta side state in memory.
        if rope_deltas is not None and prompt_kwargs.get("rope_deltas") is None:
            prompt_kwargs["rope_deltas"] = rope_deltas

        return prompt_kwargs

    @staticmethod
    def _build_batcher_decode_state(
        rope_deltas: Optional[Any],
    ) -> Optional[dict[str, Any]]:
        if rope_deltas is None:
            return None
        return {"rope_deltas": rope_deltas}

    def _enqueue_pending_spill_save(self, pending_save: PendingPromptCacheSave) -> None:
        if self._background_actor is None:
            return
        self._background_actor.enqueue_save(pending_save)

    def _prepare_request_for_insert(
        self, request: _GenerationRequest
    ) -> _PreparedInsert:
        prepared_prompt = self._prepare_prompt_inputs(request)
        restored = (
            None
            if self._prompt_cache_coordinator is None
            else self._prompt_cache_coordinator.restore(
                prompt_input_ids=prepared_prompt.prompt_input_ids,
                image_spans=prepared_prompt.image_spans,
            )
        )
        return _PreparedInsert(
            request=request,
            prepared_prompt=prepared_prompt,
            restored=restored,
        )

    def _request_needs_background_prepare(self, request: _GenerationRequest) -> bool:
        return bool(request.images_b64) or self._prompt_cache_coordinator is not None

    def _insert_prepared_request(
        self,
        batch_generator: LocalVlmBatchGenerator,
        prepared_insert: _PreparedInsert,
        active: dict[int, dict],
    ) -> None:
        request = prepared_insert.request
        prepared_prompt = prepared_insert.prepared_prompt
        restored = prepared_insert.restored
        full_prompt_input_ids = prepared_prompt.prompt_input_ids
        prompt_token_count = len(full_prompt_input_ids)

        prompt_input_ids = full_prompt_input_ids
        prompt_kwargs = None
        insert_kwargs = {}
        prompt_progress = 0
        if restored is not None:
            prompt_input_ids = full_prompt_input_ids[restored.cached_prefix_len :]
            prompt_kwargs = self._build_cached_prompt_kwargs(
                prepared_prompt,
                restored.cached_prefix_len,
                restored.rope_deltas,
            )
            prompt_progress = restored.cached_prefix_len
            insert_kwargs = {
                "caches": [restored.prompt_cache],
                "all_tokens": [full_prompt_input_ids[: restored.cached_prefix_len]],
                "decode_states": [
                    self._build_batcher_decode_state(restored.rope_deltas)
                ],
            }
        else:
            prompt_kwargs = self._build_prompt_kwargs(prepared_prompt)

        if self._prompt_cache_coordinator is not None:
            cache_boundaries = self._prompt_cache_coordinator.boundaries_after(
                prompt_input_ids=full_prompt_input_ids,
                image_spans=prepared_prompt.image_spans,
                prompt_progress=prompt_progress,
                max_prefix_len=prompt_token_count + request.max_tokens,
            )
            if cache_boundaries:
                insert_kwargs["cache_boundaries"] = [cache_boundaries]
                insert_kwargs["prompt_cache_boundary_callback"] = (
                    self._prompt_cache_coordinator.make_boundary_callback(
                        prompt_input_ids=full_prompt_input_ids,
                        image_spans=prepared_prompt.image_spans,
                    )
                )

        request.rqueue.put((prompt_progress, prompt_token_count))
        (uid,) = batch_generator.insert(
            [prompt_input_ids],
            max_tokens=[request.max_tokens],
            **insert_kwargs,
            prompt_kwargs=[prompt_kwargs],
        )

        request.rqueue.put((prompt_token_count, prompt_token_count))
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        active[uid] = {
            "rqueue": request.rqueue,
            "detokenizer": detokenizer,
            "top_logprobs": request.top_logprobs,
            "request_id": request.request_id,
            "image_spans": prepared_prompt.image_spans,
        }

    def _cancel_request(
        self,
        request_id: str,
        state: _SchedulerState,
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
            if result["request_id"] != request_id:
                continue
            if state.batch_generator is not None:
                state.batch_generator.remove(uid)
            result["rqueue"].put(RequestCancelled())
            del state.active[uid]
            return True

        return False

    def _emit_response(self, result: dict, response) -> BatchedGenerationResponse:
        detokenizer = result["detokenizer"]
        if response.finish_reason != "stop":
            detokenizer.add_token(response.token)
        if response.finish_reason is not None:
            detokenizer.finalize()

        top_logprobs = None
        if result["top_logprobs"] > 0 and response.top_logprobs is not None:
            top_logprobs = [
                Token(
                    id=int(token_id),
                    text=self.tokenizer.decode(token_id),
                    logprob=float(logprob),
                )
                for token_id, logprob in response.top_logprobs[: result["top_logprobs"]]
            ]

        return BatchedGenerationResponse(
            text=detokenizer.last_segment,
            token=int(response.token),
            token_logprob=float(response.token_logprob),
            top_logprobs=top_logprobs,
            finish_reason=response.finish_reason,
            from_draft=False,
        )

    def _drain_incoming(self, timeout: Optional[float]) -> list:
        return self._drain_queue(self._requests, timeout)

    @staticmethod
    def _drain_queue(queue: Queue, timeout: Optional[float]) -> list:
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
        item: _PreparedInsert | _FailedRestore,
        state: _SchedulerState,
    ) -> None:
        request_id = item.request.request_id
        state.restoring.pop(request_id, None)

        if request_id in state.cancelled_restores:
            state.cancelled_restores.discard(request_id)
            return

        if isinstance(item, _FailedRestore):
            item.request.rqueue.put(item.error)
            return

        state.ready.append(item)

    @staticmethod
    def _reserved_slots(state: _SchedulerState) -> int:
        return len(state.active) + len(state.ready) + len(state.restoring)

    def _drain_scheduler_events(self, state: _SchedulerState, timeout: float | None):
        for item in self._drain_incoming(timeout):
            if isinstance(item, CancelGenerationRequest):
                if not self._cancel_request(item.request_id, state):
                    logger.warning(f"Could not cancel request_id={item.request_id}")
                continue

            if isinstance(item, (_PreparedInsert, _FailedRestore)):
                self._handle_prepared_event(item, state)
                continue

            state.pending.append(item)

    def _ensure_batch_generator(self, state: _SchedulerState) -> None:
        if state.active or state.ready or state.restoring or not state.pending:
            return

        state.current_sampling_key = state.pending[0].sampling_key
        if state.batch_generator is not None:
            state.batch_generator.close()
        state.batch_generator = self._make_batch_generator(state.pending[0])

    def _insert_ready_requests(self, state: _SchedulerState) -> None:
        if state.batch_generator is None or len(state.active) >= self._max_seq_nums:
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

    def _admit_pending_requests(self, state: _SchedulerState) -> None:
        if (
            state.batch_generator is None
            or self._reserved_slots(state) >= self._max_seq_nums
        ):
            return

        next_pending = []
        for request in state.pending:
            if (
                self._reserved_slots(state) < self._max_seq_nums
                and request.sampling_key == state.current_sampling_key
            ):
                if self._request_needs_background_prepare(request):
                    if self._background_actor is None:
                        state.ready.append(self._prepare_request_for_insert(request))
                    else:
                        state.restoring[request.request_id] = request
                        self._background_actor.enqueue_restore(request)
                else:
                    state.ready.append(self._prepare_request_for_insert(request))
            else:
                next_pending.append(request)
        state.pending = next_pending

    def _step_generation(self, state: _SchedulerState) -> None:
        if not state.active or state.batch_generator is None:
            return

        for response in state.batch_generator.next():
            result = state.active.get(response.uid)
            if result is None:
                continue

            result["rqueue"].put(self._emit_response(result, response))
            if response.finish_reason is not None:
                if (
                    self._prompt_cache_coordinator is not None
                    and response.prompt_cache is not None
                    and response.all_tokens is not None
                ):
                    self._prompt_spill_cache.finalize_cap_from_completed_cache(
                        response.prompt_cache
                    )
                    self._prompt_cache_coordinator.remember_completed(
                        prompt_input_ids=response.all_tokens,
                        image_spans=result["image_spans"],
                        prompt_cache=response.prompt_cache,
                        rope_deltas=(
                            None
                            if response.decode_state is None
                            else response.decode_state.get("rope_deltas")
                        ),
                    )
                result["rqueue"].put(None)
                del state.active[response.uid]

    def _cancel_scheduler_state(self, state: _SchedulerState) -> None:
        if state.batch_generator is not None:
            state.batch_generator.close()

        for result in state.active.values():
            result["rqueue"].put(RequestCancelled("Model shutdown requested"))
        for request in state.pending:
            request.rqueue.put(RequestCancelled("Model shutdown requested"))
        for prepared_insert in state.ready:
            prepared_insert.request.rqueue.put(
                RequestCancelled("Model shutdown requested")
            )
        for request in state.restoring.values():
            request.rqueue.put(RequestCancelled("Model shutdown requested"))

    def _generate(self):
        state = _SchedulerState()

        while not self._shutdown.is_set():
            timeout = None if state.active or state.ready else 0.1
            self._drain_scheduler_events(state, timeout)
            self._ensure_batch_generator(state)
            self._insert_ready_requests(state)
            self._admit_pending_requests(state)
            self._step_generation(state)

        self._cancel_scheduler_state(state)

    def __del__(self):
        self.shutdown()
