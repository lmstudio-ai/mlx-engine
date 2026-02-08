from threading import Thread, Event
from typing import Any, List, Optional, Tuple
import json
import sys
import traceback
import mlx_lm
import logging
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
import mlx.nn as nn
import mlx.core as mx
from pathlib import Path
from queue import Queue
from queue import Empty as QueueEmpty
import time
from mlx_lm.server import LRUPromptCache
from mlx_engine.recurrent_checkpoint_store import RecurrentCheckpointStore
from mlx_engine.utils.token import Token

from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    RequestCancelled,
    GenerationRequest,
    CancelGenerationRequest,
)

logger = logging.getLogger(__name__)


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

        # Detect if model needs checkpoint-based prefix caching (hybrid/recurrent models)
        test_cache = make_prompt_cache(self.model)
        self._needs_checkpointing = not can_trim_prompt_cache(test_cache)
        self._checkpoint_store: Optional[RecurrentCheckpointStore] = (
            RecurrentCheckpointStore(max_checkpoints=16)
            if self._needs_checkpointing
            else None
        )
        del test_cache
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

        def _inner():
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

    def _prefill_with_checkpoints(
        self,
        cache: List[Any],
        tokens: mx.array,
        full_prompt_tokens: mx.array,
        rqueue: Queue,
        cached_tokens_before: int,
        chunk_size: int = 512,
    ) -> None:
        """
        Manual chunked prefill that saves checkpoints to the checkpoint store.
        Reports progress via rqueue as (processed, total) tuples.

        Args:
            cache: The model cache to fill.
            tokens: Tokens to process through the model (the remaining portion).
            full_prompt_tokens: The complete prompt token sequence (for checkpoint keys).
            rqueue: Response queue for progress reporting.
            cached_tokens_before: Number of tokens already in the cache before this prefill.
            chunk_size: Number of tokens to process per chunk.
        """
        remaining_tokens = tokens
        num_processed = 0
        total_prompt_tokens = len(full_prompt_tokens)
        cached_before = cached_tokens_before

        while remaining_tokens.size > 0:
            current_chunk_size = min(chunk_size, remaining_tokens.size)
            current_chunk = remaining_tokens[:current_chunk_size]

            self.model(current_chunk[None], cache=cache)
            mx.eval([c.state for c in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            mx.clear_cache()

            # Save checkpoint after each chunk (when more chunks remain)
            if remaining_tokens.size > 0 and self._checkpoint_store is not None:
                tokens_in_cache = cached_before + num_processed
                self._checkpoint_store.save(full_prompt_tokens[:tokens_in_cache], cache)

            # Report progress
            rqueue.put((min(cached_before + num_processed, total_prompt_tokens), total_prompt_tokens))

    def _fetch_cache_with_checkpoint_fallback(
        self,
        model_key: str,
        prompt_tokens: list,
        rqueue: Queue,
    ) -> Tuple[Any, list]:
        """
        Fetch cache from LRUPromptCache, falling back to checkpoint store for
        non-trimmable (hybrid/recurrent) models when LRUPromptCache can't help.

        Args:
            model_key: The model key for LRUPromptCache.
            prompt_tokens: The full prompt token list.
            rqueue: Response queue for progress reporting during manual prefill.

        Returns:
            Tuple of (cache, remaining_tokens) ready for batch_generator.insert().
        """
        cache, rest = self._prompt_cache.fetch_nearest_cache(model_key, prompt_tokens)

        # If LRUPromptCache found a match, use it (works for exact match and prefix extension)
        if cache is not None:
            return cache, rest

        # cache is None means no match at all. For standard models, let BatchGenerator handle prefill.
        if not self._needs_checkpointing or self._checkpoint_store is None:
            return None, prompt_tokens

        # Non-trimmable model with no LRU cache match — try checkpoint store
        # Reserve the last token for BatchGenerator (it needs at least 1 token to process)
        tokens_array = mx.array(prompt_tokens)
        result = self._checkpoint_store.find_longest_prefix(tokens_array)

        if result is not None:
            prefix_len, restored_cache = result
            remaining = tokens_array[prefix_len:]

            if remaining.size == 0:
                # Exact match — should not happen since checkpoint keys exclude the last token,
                # but guard defensively. Let BatchGenerator handle the last token.
                logger.warning("Checkpoint exact match — returning last token for BatchGenerator")
                return restored_cache, prompt_tokens[-1:]

            # Prefill all but the last token — BatchGenerator needs >= 1 token
            tokens_to_prefill = remaining[:-1] if remaining.size > 1 else mx.array([], dtype=remaining.dtype)

            if tokens_to_prefill.size > 0:
                self._prefill_with_checkpoints(
                    cache=restored_cache,
                    tokens=tokens_to_prefill,
                    full_prompt_tokens=tokens_array,
                    rqueue=rqueue,
                    cached_tokens_before=prefix_len,
                )

            # Save a checkpoint at the end of the prefill
            if prefix_len + tokens_to_prefill.size > 0:
                self._checkpoint_store.save(
                    tokens_array[: prefix_len + tokens_to_prefill.size], restored_cache
                )

            # Return cache with last token(s) for BatchGenerator
            return restored_cache, prompt_tokens[-1:]
        else:
            # No checkpoint match — let BatchGenerator handle prefill natively
            # (manual prefill here would serialize all requests, killing concurrency)
            # Checkpoints will be saved after generation completes (see _generate loop)
            return None, prompt_tokens

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

        def progress_callback(info):
            for uid, processed, total in info:
                if uid in self._batch_results:
                    self._batch_results[uid]["rqueue"].put(
                        (min(processed, total), total)
                    )

        batch_generator = BatchGenerator(
            self.model,
            max_tokens=10000000,
            completion_batch_size=self._max_seq_nums,
            # As soon as we receive any prompt, stop decoding, prefill the new prompt, and add it to the decoding batch
            # We probably want to make this behavior configurable, so that new prompts do not pause existing decodes
            prefill_batch_size=1,
            stop_tokens=set(self.tokenizer.eos_token_ids),
            # Do not set any global post-processors, sampler and logits_processor are set per-request
            sampler=None,
            logits_processors=None,
            prompt_progress_callback=progress_callback,
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

                # Get cache (with checkpoint fallback for hybrid/recurrent models)
                cache, rest = self._fetch_cache_with_checkpoint_fallback(
                    current_model_key, request.prompt_tokens, request.rqueue
                )

                # Add to batch
                (uid,) = batch_generator.insert(
                    [rest],
                    [request.max_tokens],
                    caches=[cache],
                    samplers=[request.samplers],
                    logits_processors=[request.logits_processors],
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

                responses = batch_generator.next()
                if not responses:
                    break

                for r in responses:
                    # Create response object
                    result = self._batch_results[r.uid]
                    result["cache_key"].append(r.token)
                    if r.finish_reason != "stop":
                        result["detokenizer"].add_token(r.token)
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
                            text=result["detokenizer"].last_segment,
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
                        # Save checkpoint for hybrid/recurrent models so subsequent
                        # requests (e.g. next turn in a conversation) can restore from it
                        if self._checkpoint_store is not None:
                            self._checkpoint_store.save(
                                mx.array(result["cache_key"]), r.prompt_cache
                            )
                        del self._batch_results[r.uid]

        for entry in self._batch_results.values():
            entry["rqueue"].put(RequestCancelled("Model shutdown requested"))

    def __del__(self):
        self.shutdown()
