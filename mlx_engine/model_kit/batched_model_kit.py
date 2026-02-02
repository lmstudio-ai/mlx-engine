from threading import Thread
from dataclasses import dataclass
import json
import traceback
import mlx_lm
import logging
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_lm.generate import BatchGenerator
import mlx.nn as nn
import mlx.core as mx
from pathlib import Path
from queue import Queue
from queue import Empty as QueueEmpty
from mlx_lm.models.cache import (
    make_prompt_cache,
)
import time
from mlx_lm.server import LRUPromptCache
from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_engine.utils.token import Token

logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    """Response object for batched generation, containing computed logprobs."""

    text: str
    token: int
    token_logprob: float
    top_logprobs: list[Token] | None
    finish_reason: str | None
    from_draft: bool = False


@dataclass
class GenerationRequest:
    rqueue: Queue
    prompt_tokens: list[int]
    request_id: str
    samplers: object
    logits_processors: list
    top_logprobs: int
    max_tokens: int


@dataclass
class CancelGenerationRequest:
    request_id: str


class BatchedModelKit:
    model: nn.Module
    tokenizer: TokenizerWrapper
    detokenizer: StreamingDetokenizer
    model_type: str | None
    max_kv_size: int | None
    kv_bits: int | None
    kv_group_size: int | None
    quantized_kv_start: int | None
    _generation_thread: Thread
    _requests = Queue()
    _prompt_cache = LRUPromptCache()
    _batch_results = {}
    _backend_exception: Exception | None = None
    _shutdown = False

    def __init__(
        self,
        model_path: Path,
        # vocab_only: bool = False,
        max_kv_size: int | None = None,
        kv_bits: int | None = None,
        kv_group_size: int | None = None,
        quantized_kv_start: int | None = None,
    ):
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )
        if kv_bits and max_kv_size is not None:
            # Quantized KV cache is only supported for non-rotating KV cache
            logger.warning("max_kv_size is ignored when using KV cache quantization")
            max_kv_size = None

        self.model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        config_json = json.loads((model_path / "config.json").read_text())
        self.model_type = config_json.get("model_type", None)

        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path, lazy=False)
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer, model_path=model_path, model_type=self.model_type
        )
        self.detokenizer = self.tokenizer.detokenizer
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.max_kv_size = max_kv_size
        logger.info("BatchedModelKit loaded successfully")

        mx.synchronize()  # Defensively sync before launching a new thread
        self._generation_thread = Thread(target=self._generate_with_exception_handling)
        self._generation_thread.start()

    def tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def is_cross_prompt_cache_active(self) -> bool:
        """Batched backend handles caching internally."""
        return False

    def generate(
        self,
        *,
        prompt_tokens,
        request_id,
        sampler,
        logits_processors,
        prompt_progress_callback,
        top_logprobs,
        max_tokens,
    ):
        # Do not accept new requests if error or shutdown
        if self._shutdown:
            raise RuntimeError("Cannot accept new requests when model is shutdown")
        if isinstance(self._backend_exception, Exception):
            raise self._backend_exception

        response_queue = Queue()
        self._requests.put(
            GenerationRequest(
                response_queue,
                prompt_tokens,
                request_id,
                sampler,
                logits_processors,
                top_logprobs,
                max_tokens,
            )
        )

        def _inner():
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
        self._shutdown = True
        if self._generation_thread:
            self._generation_thread.join()
        for entry in self._batch_results.values():
            entry["rqueue"].put(Exception("Model shutdown requested"))

    def _generate_with_exception_handling(self):
        try:
            self._generate()
        except Exception:
            err_string = f"Encountered fatal exception in the backend scheduler: {traceback.format_exc()}"
            logger.error(err_string)
            self._backend_exception = Exception(err_string)
            for entry in self._batch_results.values():
                entry["rqueue"].put(self._backend_exception)

    def _generate(self):
        def progress_callback(info):
            for uid, processed, total in info:
                if uid in self._batch_results:
                    self._batch_results[uid]["rqueue"].put(
                        (min(processed, total), total)
                    )

        batch_generator = BatchGenerator(
            self.model,
            max_tokens=10000000,
            stop_tokens=set(self.tokenizer.eos_token_ids),
            sampler=None,
            logits_processors=None,
            prompt_progress_callback=progress_callback,
            # TODO(christian): uncomment after merge of https://github.com/ml-explore/mlx-lm/pull/834
            # max_kv_size=self.max_kv_size,
        )
        # only using one model, so model key name value does not matter
        current_model_key = "key"

        def get_next_request(timeout=None):
            try:
                if timeout is not None:
                    return self._requests.get(timeout=timeout)
                else:
                    return self._requests.get_nowait()
            except QueueEmpty:
                return None

        while not self._shutdown:
            request = None
            timeout: None | float = None if (len(self._batch_results) > 0) else 0.1
            request = get_next_request(timeout=timeout)

            # We got a request
            if request is not None:
                if isinstance(request, CancelGenerationRequest):
                    found_request_id = False
                    request_id = request.request_id
                    for uid, entry in self._batch_results.items():
                        if entry.get("request_id") == request_id:
                            found_request_id = True
                            batch_generator.remove([uid])
                            del self._batch_results[uid]
                            break
                    if not found_request_id:
                        logger.warning(f"Could not cancel {request_id=} (id not found)")
                    continue

                cache, rest = self._prompt_cache.fetch_nearest_cache(
                    current_model_key, request.prompt_tokens
                )
                if cache is None:
                    cache = make_prompt_cache(self.model)

                (uid,) = batch_generator.insert(
                    [rest],
                    [request.max_tokens],
                    caches=[cache],
                    samplers=[request.samplers],
                    logits_processors=[request.logits_processors],
                )
                self._batch_results[uid] = {
                    # "ctx": ctx,
                    "cache_key": request.prompt_tokens[:],
                    "rqueue": request.rqueue,
                    "detokenizer": self.tokenizer.detokenizer,
                    "top_logprobs": request.top_logprobs,
                    "request_id": request.request_id,
                }
                continue

            # No request so serve from the current batch
            elif batch_generator is not None:
                if len(self._batch_results) == 0:
                    continue

                uids_to_remove = []
                time_budget = 0.5
                start = time.time()
                while True:
                    if time.time() - start > time_budget:
                        break

                    responses = batch_generator.next()
                    if not responses:
                        break

                    for r in responses:
                        result = self._batch_results[r.uid]
                        result["cache_key"].append(r.token)
                        if r.finish_reason != "stop":
                            result["detokenizer"].add_token(r.token)

                        token_logprob = r.logprobs[r.token].item()
                        top_logprobs_list = None
                        if result["top_logprobs"] > 0:
                            sorted_indices = mx.argpartition(
                                -r.logprobs, kth=result["top_logprobs"] - 1
                            )
                            top_indices = sorted_indices[: result["top_logprobs"]]
                            top_logprobs_values = r.logprobs[top_indices]

                            top_logprobs_list = [
                                Token(
                                    id=int(idx),
                                    text=self.tokenizer.decode(idx),
                                    logprob=float(prob),
                                )
                                for idx, prob in zip(
                                    top_indices.tolist(), top_logprobs_values.tolist()
                                )
                            ]

                        result["rqueue"].put(
                            GenerationResponse(
                                text=result["detokenizer"].last_segment,
                                token=r.token,
                                token_logprob=token_logprob,
                                top_logprobs=top_logprobs_list,
                                finish_reason=r.finish_reason,
                                from_draft=False,
                            )
                        )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self._prompt_cache.insert_cache(
                                current_model_key, result["cache_key"], r.prompt_cache
                            )
                            del self._batch_results[r.uid]

                        # if result["ctx"]._should_stop:
                        #     uids_to_remove.append(r.uid)

                    if uids_to_remove:
                        batch_generator.remove(uids_to_remove)

    def __del__(self):
        self.shutdown()
