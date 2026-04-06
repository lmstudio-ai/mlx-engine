from threading import Thread, Event
import json
import sys
import traceback
from typing import Iterable
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
import time
from mlx_lm.server import LRUPromptCache
from mlx_engine.utils.token import Token

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    RequestCancelled,
    GenerationRequest,
    CancelGenerationRequest,
)

logger = logging.getLogger(__name__)


def _make_batched_logits_processor(processor, state, trim_prefix_len):
    def wrapped(tokens, logits):
        # Newer mlx-lm batch generation passes Python token lists here,
        # while our processors still expect an MLX array.
        if not hasattr(tokens, "shape"):
            tokens = mx.array(tokens)

        if trim_prefix_len > 0:
            tokens = tokens[trim_prefix_len:]

        # In batched mlx-lm, logits processors are called before the current
        # input token has been appended to `tokens`. Restore the sequential-path
        # view by appending the token that was just fed into the model.
        pending_input_tokens = state["pending_input_tokens"]
        if pending_input_tokens is not None and pending_input_tokens.size > 0:
            tokens = mx.concatenate([tokens, pending_input_tokens])

        return processor(tokens, logits)

    return wrapped


def _wrap_batched_sampler(sampler, state):
    if sampler is None:
        return None

    def wrapped(logprobs):
        sampled = sampler(logprobs)
        state["pending_input_tokens"] = mx.array(sampled).reshape(-1)
        return sampled

    return wrapped


def _wrap_batched_logits_processors(logits_processors, state, trim_prefix_len=0):
    return [
        _make_batched_logits_processor(processor, state, trim_prefix_len)
        for processor in (logits_processors or [])
    ]


class BatchedModelKit:
    """
    This model kit enables continuous batching by running `mlx_lm.BatchGenerator` in a worker thread

    Args:
        model_path (Path): Path to the model directory containing model files.
        max_kv_size (Optional[int]): Maximum size of the key-value cache.
        max_seq_nums (Optional[int]): Maximum number of concurrent generation requests
            that can be processed simultaneously.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    model_type: str | None

    _detokenizer: StreamingDetokenizer
    _model_path: Path
    _max_kv_size: int | None
    _max_seq_nums: int
    _generation_thread: Thread
    _requests: Queue
    _prompt_cache: LRUPromptCache
    _batch_results: dict
    _backend_exception: Exception | None
    _shutdown: Event

    def __init__(
        self,
        model_path: Path,
        prefill_step_size: int,
        max_kv_size: int | None = None,
        max_seq_nums: int | None = None,
    ):
        self._requests = Queue()
        self._prompt_cache = LRUPromptCache()
        self._batch_results = {}
        self._backend_exception = None
        self._generation_thread = None
        self._shutdown = Event()
        if max_seq_nums is None or max_seq_nums < 1:
            max_seq_nums = 1
            logger.info(f"Setting concurrent request limit to {max_seq_nums}")
        self._max_seq_nums = max_seq_nums

        self._model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        config_json = json.loads((model_path / "config.json").read_text())
        self.model_type = config_json.get("model_type", None)

        self.model, self.tokenizer = mlx_lm.utils.load(self._model_path, lazy=False)
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer, model_path=model_path, model_type=self.model_type
        )
        self._detokenizer = self.tokenizer.detokenizer
        self._max_kv_size = max_kv_size
        self._prefill_step_size = prefill_step_size
        logger.info("BatchedModelKit loaded successfully")

    def start(self):
        """
        Start the background generation thread.
        """
        mx.synchronize()  # Defensively sync before launching a new thread
        # Set daemon flag, which tells python it's okay to exit if this is the only alive thread at exit.
        self._generation_thread = Thread(
            target=self._generate_with_exception_handling, daemon=True
        )
        self._generation_thread.start()

    def tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def is_cross_prompt_cache_active(self) -> bool:
        """
        Check if cross-prompt caching is enabled.

        Returns:
            bool: Always False for BatchedModelKit as it handles caching internally
        """
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
        """
        This method queues a generation request to the background thread and returns
        an iterator that yields BatchedGenerationResponse objects as tokens are generated.

        Args:
            prompt_tokens: List of token IDs representing the input prompt
            request_id: Unique identifier for this generation request
            sampler: Sampling function for token selection
            logits_processors: List of logits processors to apply
            prompt_progress_callback: Callback for reporting prompt processing progress
            top_logprobs: Number of top token probabilities to return
            max_tokens: Maximum number of tokens to generate

        Yields:
            BatchedGenerationResponse: Generated tokens with their probabilities
            Raises Exceptions when generation needs to be stopped

        Raises:
            RuntimeError: If model is shutdown or an exception occurred in the backend
        """
        # Do not accept new requests if error or shutdown
        if self._shutdown.is_set():
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

        def _inner() -> Iterable[BatchedGenerationResponse]:
            """
            Generator that pulls responses from the queue and yields them to the caller.

            Handles three types of responses from the background thread:
            1. None: Signals generation completion
            2. Exception: Re-raises exceptions from the background thread
            3. Tuple: Prompt progress updates (processed, total)
            4. BatchedGenerationResponse: Generated tokens to yield
            """
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
        """
        Cancel a generation request by its request ID.

        Queues a cancellation request to the background thread. The request will be
        removed from the batch and its response queue will receive a RequestCancelled
        exception.
        """
        self._requests.put(CancelGenerationRequest(request_id))

    def shutdown(self):
        """
        Shutdown the background generation thread and clean up resources.

        Sets the shutdown flag and waits for the generation thread to complete.
        All pending requests will receive a RequestCancelled exception.
        """
        if not self._shutdown.is_set():
            self._shutdown.set()
            if self._generation_thread:
                self._generation_thread.join()

    def _generate_with_exception_handling(self):
        """
        Wrapper around _generate that catches and handles fatal exceptions.

        If an exception occurs during generation, it logs the error, stores it
        in _backend_exception, sends it to all active request queues, and exits
        the process after a brief delay.
        """
        try:
            self._generate()
        except Exception:
            err_string = f"Encountered fatal exception in the backend scheduler: {traceback.format_exc()}"
            logger.error(err_string)
            # Cancel ongoing prediction threads
            self._backend_exception = Exception(err_string)
            for entry in self._batch_results.values():
                entry["rqueue"].put(self._backend_exception)

            # Sleep to allow error messages to be logged and propagated to clients
            time.sleep(3)

            # Exit with non-zero code because the backend scheduler is dead and cannot recover.
            # Continuing would leave the process in a broken state where new requests hang indefinitely.
            sys.exit(1)

    def _generate(self):
        """
        Main generation loop running in the background thread.

        This method continuously processes requests from the queue:
        1. Accepts new generation requests and adds them to the batch
        2. Handles cancellation requests by removing requests from the batch
        3. Generates tokens for all active requests in the batch
        4. Sends generated tokens back through each request's response queue
        5. Manages prompt caching for efficient reuse of common prefixes

        The loop runs until shutdown is requested, at which point all active
        requests receive a RequestCancelled exception.
        """

        batch_generator = BatchGenerator(
            self.model,
            max_tokens=10000000,
            completion_batch_size=self._max_seq_nums,
            # As soon as we receive any prompt, stop decoding, prefill the new prompt, and add it to the decoding batch
            # We probably want to make this behavior configurable, so that new prompts do not pause existing decodes
            prefill_batch_size=1,
            prefill_step_size=self._prefill_step_size,
            stop_tokens=[[token] for token in self.tokenizer.eos_token_ids],
            # Do not set any global post-processors, sampler and logits_processor are set per-request
            sampler=None,
            logits_processors=None,
            max_kv_size=self._max_kv_size,
        )
        # only using one model, so model key name value does not matter
        current_model_key = "lmstudio"

        def get_next_request(timeout=None):
            try:
                if timeout is not None:
                    return self._requests.get(timeout=timeout)
                else:
                    return self._requests.get_nowait()
            except QueueEmpty:
                return None

        while not self._shutdown.is_set():
            request = None
            timeout: None | float = None if (len(self._batch_results) > 0) else 0.1
            request = get_next_request(timeout=timeout)

            # We got a request
            if request is not None:
                if isinstance(request, CancelGenerationRequest):
                    # Handle cancel request
                    found_request_id = False
                    request_id = request.request_id
                    for uid, entry in self._batch_results.items():
                        if entry.get("request_id") == request_id:
                            found_request_id = True
                            batch_generator.remove([uid])
                            self._batch_results[uid]["rqueue"].put(RequestCancelled())
                            del self._batch_results[uid]
                            break
                    if not found_request_id:
                        logger.warning(f"Could not cancel {request_id=} (id not found)")
                    continue

                # Get cache
                cache, rest = self._prompt_cache.fetch_nearest_cache(
                    current_model_key, request.prompt_tokens
                )
                initial_tokens = rest[-1:]
                trim_prefix_len = max(len(rest) - len(initial_tokens), 0)
                state = {
                    "pending_input_tokens": (
                        mx.array(initial_tokens) if initial_tokens else None
                    )
                }

                # Add to batch
                (uid,) = batch_generator.insert(
                    [rest],
                    [request.max_tokens],
                    caches=[cache],
                    samplers=[_wrap_batched_sampler(request.samplers, state)],
                    logits_processors=[
                        _wrap_batched_logits_processors(
                            request.logits_processors,
                            state=state,
                            trim_prefix_len=trim_prefix_len,
                        )
                    ],
                )

                # Track this request
                self._batch_results[uid] = {
                    "cache_key": request.prompt_tokens[:],
                    "rqueue": request.rqueue,
                    "detokenizer": self.tokenizer.detokenizer,
                    "top_logprobs": request.top_logprobs,
                    "request_id": request.request_id,
                }

                # Check for new requests
                continue

            # No request so serve from the current batch
            if len(self._batch_results) == 0:
                continue

            time_budget = 0.5
            start = time.time()
            while True:
                if time.time() - start > time_budget:
                    break

                prompt_responses, generation_responses = batch_generator.next()
                if not prompt_responses and not generation_responses:
                    break

                for r in prompt_responses:
                    result = self._batch_results.get(r.uid)
                    if result is not None:
                        processed, total = r.progress
                        result["rqueue"].put((min(processed, total), total))

                for r in generation_responses:
                    # Create response object
                    result = self._batch_results[r.uid]
                    detokenizer = result["detokenizer"]
                    result["cache_key"].append(r.token)
                    if r.finish_reason != "stop":
                        detokenizer.add_token(r.token)
                    if r.finish_reason is not None:
                        detokenizer.finalize()
                    token_logprob = r.logprobs[r.token].item()
                    top_logprobs_list = None

                    # Ensure MLX-based logprob processing happens in this MLX worker thread
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

                    # Send the result
                    result["rqueue"].put(
                        BatchedGenerationResponse(
                            text=detokenizer.last_segment,
                            token=r.token,
                            token_logprob=token_logprob,
                            top_logprobs=top_logprobs_list,
                            finish_reason=r.finish_reason,
                            from_draft=False,
                        )
                    )

                    # Clean up if necessary
                    if r.finish_reason is not None:
                        result["rqueue"].put(None)
                        self._prompt_cache.insert_cache(
                            current_model_key, result["cache_key"], r.prompt_cache
                        )
                        del self._batch_results[r.uid]

        for entry in self._batch_results.values():
            entry["rqueue"].put(RequestCancelled("Model shutdown requested"))

    def __del__(self):
        self.shutdown()
