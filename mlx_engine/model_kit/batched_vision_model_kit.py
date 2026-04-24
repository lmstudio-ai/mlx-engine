import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Thread
from typing import Iterable, Optional

import mlx.core as mx
import mlx_lm
import mlx_vlm

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.utils.generation_helpers import MAX_TOP_LOGPROBS, create_sampler
from mlx_engine.utils.image_utils import convert_to_pil, custom_resize
from mlx_engine.utils.token import Token
from mlx_engine.vision_model_kit._transformers_compatibility import (
    fix_qwen2_5_vl_image_processor,
    fix_qwen2_vl_preprocessor,
)
from mlx_vlm.generate import BatchGenerator

logger = logging.getLogger(__name__)


MLX_VLM_BATCHED_VISION_ENV_VAR = "MLX_ENGINE_USE_MLX_VLM_BATCHED_VISION"


def is_mlx_vlm_batched_vision_enabled() -> bool:
    raw = os.environ.get(MLX_VLM_BATCHED_VISION_ENV_VAR, "")
    return raw.lower() in {"1", "true", "yes", "on"}


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


class BatchedVisionModelKit:
    """
    Feature-flagged VLM batching backend built on mlx-vlm's BatchGenerator.

    V1 deliberately keeps the scope narrow: a single worker thread owns the
    mlx-vlm batcher and handles both text-only and basic image requests.
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
        max_seq_nums: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
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

        if vocab_only:
            self._init_tokenizer_only()
        else:
            self._init_full_model()

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

    def _init_full_model(self) -> None:
        return_tuple = mlx_vlm.utils.load(
            self._model_path,
            processor_config={"trust_remote_code": self._trust_remote_code},
            trust_remote_code=self._trust_remote_code,
        )
        if len(return_tuple) == 2:
            self.model, self.processor = return_tuple
        else:
            self.model, self.processor, _ = return_tuple

        self.tokenizer = mlx_lm.tokenizer_utils.load(
            self._model_path, eos_token_ids=self._get_eos_token_ids()
        )
        self.detokenizer = self.tokenizer.detokenizer
        mx.clear_cache()

    def start(self):
        if self.vocab_only:
            return
        mx.synchronize()
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
            self._generate()
        except Exception:
            err_string = f"Encountered fatal exception in the backend scheduler: {traceback.format_exc()}"
            logger.error(err_string)
            self._backend_exception = Exception(err_string)

            # Sleep to allow error messages to be logged and propagated to clients.
            time.sleep(3)
            sys.exit(1)

    def _make_batch_generator(self, request: _GenerationRequest) -> BatchGenerator:
        sampler = create_sampler(
            request.temp,
            request.top_p,
            request.min_p,
            request.min_tokens_to_keep,
            request.top_k,
        )
        return BatchGenerator(
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

    def _prepare_prompt_kwargs(
        self, request: _GenerationRequest
    ) -> tuple[list[int], dict]:
        prompt_tokens = request.prompt_tokens
        if len(prompt_tokens) == 0:
            prompt_tokens = self.tokenize(" ")

        if not request.images_b64:
            input_ids = mx.array(prompt_tokens, dtype=mx.int32)[None, :]
            embedding_output = self.model.get_input_embeddings(input_ids)
            return input_ids.squeeze(0).tolist(), embedding_output.to_dict()

        # Keep image decode and embedding prep on the worker thread for now.
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
        return input_ids.squeeze(0).tolist(), {
            **data_kwargs,
            **embedding_output.to_dict(),
        }

    def _insert_request(
        self,
        batch_generator: BatchGenerator,
        request: _GenerationRequest,
        active: dict[int, dict],
    ) -> None:
        prompt_input_ids, prompt_kwargs = self._prepare_prompt_kwargs(request)
        prompt_token_count = len(prompt_input_ids)
        request.rqueue.put((0, prompt_token_count))
        (uid,) = batch_generator.insert(
            [prompt_input_ids],
            max_tokens=[request.max_tokens],
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
        }

    def _cancel_request(
        self,
        request_id: str,
        batch_generator: Optional[BatchGenerator],
        active: dict[int, dict],
        pending: list[_GenerationRequest],
    ) -> bool:
        for i, request in enumerate(pending):
            if request.request_id == request_id:
                pending.pop(i)
                request.rqueue.put(RequestCancelled())
                return True

        for uid, result in list(active.items()):
            if result["request_id"] != request_id:
                continue
            if batch_generator is not None:
                batch_generator.remove(uid)
            result["rqueue"].put(RequestCancelled())
            del active[uid]
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
        items = []
        try:
            item = (
                self._requests.get(timeout=timeout)
                if timeout is not None
                else self._requests.get_nowait()
            )
            items.append(item)
        except QueueEmpty:
            return items

        while True:
            try:
                items.append(self._requests.get_nowait())
            except QueueEmpty:
                return items

    def _generate(self):
        batch_generator = None
        active: dict[int, dict] = {}
        pending: list[_GenerationRequest] = []
        current_sampling_key = None

        while not self._shutdown.is_set():
            timeout = None if active else 0.1
            for item in self._drain_incoming(timeout):
                if isinstance(item, CancelGenerationRequest):
                    if not self._cancel_request(
                        item.request_id, batch_generator, active, pending
                    ):
                        logger.warning(f"Could not cancel request_id={item.request_id}")
                    continue
                pending.append(item)

            if not active and pending:
                current_sampling_key = pending[0].sampling_key
                if batch_generator is not None:
                    batch_generator.close()
                batch_generator = self._make_batch_generator(pending[0])

            if batch_generator is not None and len(active) < self._max_seq_nums:
                next_pending = []
                for request in pending:
                    if (
                        len(active) < self._max_seq_nums
                        and request.sampling_key == current_sampling_key
                    ):
                        self._insert_request(batch_generator, request, active)
                    else:
                        next_pending.append(request)
                pending = next_pending

            if not active or batch_generator is None:
                continue

            _, generation_responses = batch_generator.next()
            for response in generation_responses:
                result = active.get(response.uid)
                if result is None:
                    continue

                result["rqueue"].put(self._emit_response(result, response))
                if response.finish_reason is not None:
                    result["rqueue"].put(None)
                    del active[response.uid]

        if batch_generator is not None:
            batch_generator.close()

        for result in active.values():
            result["rqueue"].put(RequestCancelled("Model shutdown requested"))
        for request in pending:
            request.rqueue.put(RequestCancelled("Model shutdown requested"))

    def __del__(self):
        self.shutdown()
