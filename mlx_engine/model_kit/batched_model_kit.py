from typing import Callable
from threading import Thread
import json
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
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
import time
from mlx_lm.server import LRUPromptCache

logger = logging.getLogger(__name__)



class BatchedModelKit:
    model: nn.Module
    tokenizer: TokenizerWrapper
    detokenizer: StreamingDetokenizer
    model_type: str | None
    _generation_thread: Thread
    _requests = Queue()
    stop = False
    _prompt_cache = LRUPromptCache()

    def __init__(
        self,
        model_path: Path,
        # vocab_only: bool = False,
        # max_kv_size: Optional[int] = None,
        # kv_bits: Optional[int] = None,
        # kv_group_size: Optional[int] = None,
        # quantized_kv_start: Optional[int] = None,
    ):
        self.model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        config_json = json.loads((model_path / "config.json").read_text())
        self.model_type = config_json.get("model_type", None)

        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path, lazy=False)
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer, model_path=model_path, model_type=self.model_type
        )
        self.detokenizer = self.tokenizer.detokenizer
        logger.info("BatchedModelKit loaded successfully")

        mx.synchronize()
        self._generation_thread = Thread(target=self._generate)
        self._generation_thread.start()

    def tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def generate(self,
                *,
                prompt_tokens,
                sampler,
                logits_processors,
                prompt_progress_callback,
    ):
        response_queue = Queue()
        self._requests.put((response_queue, prompt_tokens, sampler, logits_processors))

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

    def _prompt_processing_callback(self):
        pass

    def _generate(self):
        def progress_callback(info):
            for uid, processed, total in info:
                if uid in batch_results:
                    batch_results[uid]["rqueue"].put((min(processed, total), total))

        batch_generator = BatchGenerator(
            self.model,
            max_tokens = 10000000,
            stop_tokens = None,
            sampler = None,
            logits_processors = None,
            prompt_progress_callback = progress_callback,
        )
        current_model_key = "key" # only using one model, so model key name value does not matter
        batch_results = {}


        def get_next_request(timeout=None):
            try:
                if timeout is not None:
                    return self._requests.get(timeout=timeout)
                else:
                    return self._requests.get_nowait()
            except QueueEmpty:
                return None

        while not self.stop:
            request = None
            timeout: None | float = (
                None
                if (len(batch_results) > 0)
                else 0.1
            )
            request = get_next_request(timeout=timeout)

            # We got a request
            if request is not None:
                rqueue, prompt, samplers, logits_processors = request

                cache, rest = self._prompt_cache.fetch_nearest_cache(
                    current_model_key, prompt
                )
                if cache is None:
                    cache = make_prompt_cache(self.model)

                (uid,) = batch_generator.insert(
                    [rest],
                    [10000000], # max tokens
                    caches=[cache],
                    samplers=[samplers],
                    logits_processors=[logits_processors],
                )
                batch_results[uid] = {
                    # "ctx": ctx,
                    "cache_key": prompt[:],
                    "rqueue": rqueue,
                    "detokenizer": self.tokenizer.detokenizer,
                }
                continue

            # No request so serve from the current batch
            elif batch_generator is not None:
                if len(batch_results) == 0:
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
                        result = batch_results[r.uid]
                        result["cache_key"].append(r.token)
                        if r.finish_reason != "stop":
                            result["detokenizer"].add_token(r.token)

                        top_tokens = None
                        # TODO: fix logprobs
                        # if args.logprobs > 0:
                        #     sorted_indices = mx.argpartition(
                        #         -r.logprobs, kth=args.logprobs - 1
                        #     )
                        #     top_indices = sorted_indices[: args.logprobs]
                        #     top_logprobs = r.logprobs[top_indices]
                        #     top_token_info = zip(
                        #         top_indices.tolist(), top_logprobs.tolist()
                        #     )
                        #     top_tokens = tuple(top_token_info)

                        # TODO: fixme
                        print(result["detokenizer"].last_segment)
                        # result["rqueue"].put(
                        #     Response(
                        #         result["detokenizer"].last_segment,
                        #         r.token,
                        #         r.logprobs[r.token].item(),
                        #         r.finish_reason,
                        #         top_tokens,
                        #     )
                        # )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self._prompt_cache.insert_cache(
                                current_model_key, result["cache_key"], r.prompt_cache
                            )
                            del batch_results[r.uid]

                        # if result["ctx"]._should_stop:
                        #     uids_to_remove.append(r.uid)

                    if uids_to_remove:
                        batch_generator.remove(uids_to_remove)
