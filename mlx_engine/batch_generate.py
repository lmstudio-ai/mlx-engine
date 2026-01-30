from collections import deque
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)
from mlx_engine.utils.token import Token
from mlx_engine.utils.top_logprobs import summarize_top_logprobs
from mlx_engine.stop_string_processor import (
    StopStringProcessor,
    StopStringProcessorResult,
)
from mlx_engine.utils.set_seed import set_seed
from outlines.processors.structured import JSONLogitsProcessor
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.generate import GenerationStopCondition, MAX_TOP_LOGPROBS


EPHEMERAL_SESSION_PREFIX = "__ephemeral_session/"


@dataclass
class BatchGenerationResult:
    request_slot_id: int
    text: str
    tokens: List[Token]
    top_logprobs: List[List[Token]]
    stop_condition: Optional[GenerationStopCondition]


class PromptCacheStore:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        count: int

    @dataclass
    class SearchResult:
        exact: Optional[List[int]]
        shorter: Optional[List[int]]
        longer: Optional[List[int]]
        common_prefix: int

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._cache_by_session: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._least_recently_used: deque[Tuple[str, List[int]]] = deque()

    def _search(self, session_id: str, tokens: List[int]) -> SearchResult:
        if session_id not in self._cache_by_session:
            return self.SearchResult(None, None, None, 0)

        current_node = self._cache_by_session[session_id]
        last_cache_index = -1
        token_index = 0

        while token_index < len(tokens) and tokens[token_index] in current_node:
            current_node = current_node[tokens[token_index]]
            if "cache" in current_node:
                last_cache_index = token_index
            token_index += 1

        if last_cache_index == len(tokens) - 1:
            return self.SearchResult(tokens, None, None, 0)

        shorter = None
        if last_cache_index > 0:
            shorter = tokens[: last_cache_index + 1]

        longer = None
        common_prefix = token_index
        if token_index > 0 and last_cache_index <= 0:
            best_suffix = None
            stack: List[Tuple[Dict[str, Any], List[int]]] = [(current_node, [])]
            while len(stack) > 0:
                next_node, suffix_tokens = stack.pop()
                if "cache" in next_node:
                    if best_suffix is None or len(suffix_tokens) < len(best_suffix):
                        best_suffix = suffix_tokens
                else:
                    for token_value in next_node:
                        stack.append((next_node[token_value], suffix_tokens + [token_value]))
            if best_suffix is not None:
                longer = tokens[:token_index] + best_suffix

        return self.SearchResult(
            tokens if last_cache_index == len(tokens) - 1 else None,
            shorter,
            longer,
            common_prefix,
        )

    def _get(self, session_id: str, tokens: List[int]) -> CacheEntry:
        current_node = self._cache_by_session[session_id]
        for token_value in tokens:
            current_node = current_node[token_value]
        return current_node["cache"]

    def _delete(self, session_id: str, tokens: List[int]) -> None:
        path: List[Dict[str, Any]] = [self._cache_by_session[session_id]]
        for token_value in tokens:
            path.append(path[-1][token_value])
        del path[-1]["cache"]
        for index in reversed(range(len(tokens))):
            previous_node = path[index]
            next_node = path[index + 1]
            token_value = tokens[index]
            if len(next_node) > 0:
                break
            del previous_node[token_value]

        if len(self._cache_by_session[session_id]) == 0:
            del self._cache_by_session[session_id]

    def _extract(self, session_id: str, tokens: List[int]) -> CacheEntry:
        cache_entry = self._get(session_id, tokens)
        if cache_entry.count == 1:
            self._delete(session_id, tokens)
            self._least_recently_used.remove((session_id, tokens))
            return cache_entry

        cache_entry.count -= 1
        return self.CacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)

    def fetch_nearest_cache(
        self, session_id: str, tokens: List[int]
    ) -> Tuple[Optional[List[Any]], List[int]]:
        result = self._search(session_id, tokens)
        if result.exact is not None:
            cache_entry = self._extract(session_id, result.exact)
            return cache_entry.prompt_cache, []

        if result.shorter is not None:
            cache_entry = self._extract(session_id, result.shorter)
            prefix_length = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_length:]

        if result.longer is not None:
            cache_entry = self._get(session_id, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache_entry = self.CacheEntry(copy.deepcopy(cache_entry.prompt_cache), 1)
                prefix_length = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix_length
                trim_prompt_cache(cache_entry.prompt_cache, num_to_trim)
                return cache_entry.prompt_cache, tokens[prefix_length:]

        return None, tokens

    def insert_cache(self, session_id: str, tokens: List[int], prompt_cache: List[Any]) -> None:
        if len(tokens) == 0:
            return

        if session_id not in self._cache_by_session:
            self._cache_by_session[session_id] = {}
        current_node = self._cache_by_session[session_id]
        for token_value in tokens:
            if token_value not in current_node:
                current_node[token_value] = {}
            current_node = current_node[token_value]

        entry_key = (session_id, tokens)
        if "cache" in current_node:
            current_node["cache"].count += 1
            if entry_key in self._least_recently_used:
                self._least_recently_used.remove(entry_key)
        else:
            current_node["cache"] = self.CacheEntry(prompt_cache, 1)

        self._least_recently_used.append(entry_key)
        if len(self._least_recently_used) > self._max_size:
            oldest_session_id, oldest_tokens = self._least_recently_used.popleft()
            self._delete(oldest_session_id, oldest_tokens)


@dataclass
class BatchRequestState:
    session_id: str
    cache_key: List[int]
    cached_tokens: int
    total_prompt_tokens: int
    detokenizer: Any
    stop_string_processor: Optional[StopStringProcessor]
    token_buffer: List[Token]
    top_logprobs_buffer: List[List[Token]]
    text_buffer: str
    top_logprobs: int
    should_cache: bool


class BatchGeneratorWrapper:
    """Continuous batching wrapper around mlx-lm's BatchGenerator."""

    def __init__(
        self,
        model_kit: ModelKit,
        *,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        prompt_progress_callback: Optional[
            Callable[[List[Tuple[int, int, int]]], None]
        ] = None,
    ):
        if isinstance(model_kit, VisionModelKit):
            raise ValueError("Batching is not supported for vision models")
        # TODO(christian): it's not supported for speculative decoding, either, where to put?

        self._model_kit = model_kit
        self._tokenizer = model_kit.tokenizer
        # the underlying BatchGenerator will only start to process new prompts if
        # completion_batch_size - num_active_generations >= prefill_batch_size.
        # if prefill_batch_size != 1, this can lead to situations where continuous batching
        # appears to hang
        self._batch_generator = BatchGenerator(
            model_kit.model,
            stop_tokens=self._tokenizer.eos_token_ids,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=prompt_progress_callback,
        )
        self._request_states: Dict[int, BatchRequestState] = {}
        self._prompt_cache_store = PromptCacheStore()

    def close(self) -> None:
        """Close the batch generator and release its resources."""
        self._batch_generator.close()

    def insert(
        self,
        *,
        session_id: str,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_strings: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        temp: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        json_schema: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Insert a request into the batch generator.

        This method configures caching, sampling, and stop string processing, then
        enqueues a prompt into the underlying batch generator for batched inference.

        Args:
            session_id (str): Session identifier used for prompt caching.
            prompt_tokens (List[int]): Prompt tokens to enqueue for generation.
            max_tokens (int): Maximum number of tokens to generate.
            stop_strings (Optional[List[str]]): Strings that stop generation when matched.
            top_logprobs (Optional[int]): Number of top logprobs to include per token.
            repetition_penalty (Optional[float]): Penalty applied to repeated tokens.
            temp (Optional[float]): Sampling temperature.
            top_p (Optional[float]): Top-p (nucleus) sampling parameter.
            min_p (Optional[float]): Minimum probability threshold for sampling.
            top_k (Optional[int]): Top-k sampling parameter.
            seed (Optional[int]): Seed for deterministic sampling.
            json_schema (Optional[str]): JSON schema for structured output.

        Returns:
            Tuple[int, int, int]: The request slot id, cached token count, and total prompt tokens.
        """
        if len(prompt_tokens) == 0:
            prompt_tokens = self._model_kit.tokenize(" ")

        if seed is not None:
            set_seed(seed)

        if top_logprobs is None:
            top_logprobs = 0
        if top_logprobs > MAX_TOP_LOGPROBS:
            raise ValueError(
                f"top_logprobs must be less than or equal to {MAX_TOP_LOGPROBS}"
            )

        should_cache = session_id != "" and not session_id.startswith(
            EPHEMERAL_SESSION_PREFIX
        )
        prompt_cache = None
        remaining_prompt_tokens = prompt_tokens
        cached_tokens = 0
        total_prompt_tokens = len(prompt_tokens)
        if should_cache:
            prompt_cache, remaining_prompt_tokens = self._prompt_cache_store.fetch_nearest_cache(
                session_id, prompt_tokens
            )
            cached_tokens = total_prompt_tokens - len(remaining_prompt_tokens)

        if len(remaining_prompt_tokens) == 0:
            if prompt_cache is not None and can_trim_prompt_cache(prompt_cache):
                trim_prompt_cache(prompt_cache, 1)
                remaining_prompt_tokens = prompt_tokens[-1:]
                cached_tokens = max(0, total_prompt_tokens - 1)
            else:
                prompt_cache = None
                remaining_prompt_tokens = prompt_tokens
                cached_tokens = 0

        stop_strings = stop_strings if stop_strings is not None else []
        stop_string_processor = (
            StopStringProcessor(stop_strings, self._tokenizer)
            if len(stop_strings) > 0
            else None
        )

        logits_processors: List[Callable[[mx.array, mx.array], mx.array]] = []
        if repetition_penalty is not None and repetition_penalty != 0.0:
            if len(remaining_prompt_tokens) > 0:
                token_history = prompt_tokens[:cached_tokens]
            else:
                token_history = prompt_tokens
            logits_processors.append(
                RepetitionPenaltyProcessor(
                    token_history=token_history, repetition_penalty=repetition_penalty
                )
            )

        if json_schema is not None and json_schema != "":
            logits_processors.append(
                JSONLogitsProcessor(
                    json_schema,
                    OutlinesTransformerTokenizer(self._tokenizer._tokenizer),
                    tensor_library_name="mlx",
                )
            )

        sampler = make_sampler(
            temp=temp if temp is not None else 0.0,
            top_p=top_p if top_p is not None else 0.0,
            min_p=min_p if min_p is not None else 0.0,
            top_k=top_k if top_k is not None else 0,
        )

        insert_parameters = inspect.signature(self._batch_generator.insert).parameters
        insert_kwargs: Dict[str, Any] = {}
        cache_list = [prompt_cache]
        if "caches" in insert_parameters:
            insert_kwargs["caches"] = cache_list
        elif "cache" in insert_parameters:
            insert_kwargs["cache"] = prompt_cache

        if "samplers" in insert_parameters:
            insert_kwargs["samplers"] = [sampler]
        elif "sampler" in insert_parameters:
            insert_kwargs["sampler"] = sampler

        if "logits_processors" in insert_parameters:
            insert_kwargs["logits_processors"] = [logits_processors]

        request_slot_ids = self._batch_generator.insert(
            [remaining_prompt_tokens],
            max_tokens,
            **insert_kwargs,
        )
        request_slot_id = request_slot_ids[0]
        request_state = BatchRequestState(
            session_id=session_id,
            cache_key=list(prompt_tokens),
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            detokenizer=self._tokenizer.detokenizer,
            stop_string_processor=stop_string_processor,
            token_buffer=[],
            top_logprobs_buffer=[],
            text_buffer="",
            top_logprobs=top_logprobs,
            should_cache=should_cache,
        )
        self._request_states[request_slot_id] = request_state
        return request_slot_id, cached_tokens, total_prompt_tokens

    def remove(self, request_slot_ids: List[int]) -> None:
        """
        Remove requests by slot id from the batch generator and local state.

        This method removes request state and informs the underlying batch generator
        so that the request is no longer scheduled.

        Args:
            request_slot_ids (List[int]): Request slot ids to remove.

        Returns:
            None
        """
        if len(request_slot_ids) == 0:
            return
        self._batch_generator.remove(request_slot_ids)
        for request_slot_id in request_slot_ids:
            if request_slot_id in self._request_states:
                del self._request_states[request_slot_id]

    def next(self) -> List[BatchGenerationResult]:
        """
        Advance the batch generator and return any newly generated results.

        This method steps the underlying batch generator and assembles results for
        each request slot that produced tokens during this step.

        Args:
            None

        Returns:
            List[BatchGenerationResult]: Results generated since the last call.
        """
        results: List[BatchGenerationResult] = []
        responses = self._batch_generator.next()
        if len(responses) == 0:
            return results

        for response in responses:
            request_slot_id = response.uid
            request_state = self._request_states.get(request_slot_id)
            if request_state is None:
                continue
            result = self._process_response(request_slot_id, response, request_state)
            if result is not None:
                results.append(result)

        return results

    def _process_response(
        self,
        request_slot_id: int,
        response: BatchGenerator.Response,
        request_state: BatchRequestState,
    ) -> Optional[BatchGenerationResult]:
        token_id = response.token
        request_state.cache_key.append(token_id)

        request_state.detokenizer.add_token(token_id)
        text_segment = request_state.detokenizer.last_segment
        if text_segment != "":
            request_state.text_buffer += text_segment

        logprob_value = response.logprobs[token_id].item()
        token_entry = Token(
            id=token_id, text=self._tokenizer.decode(token_id), logprob=float(logprob_value)
        )
        request_state.token_buffer.append(token_entry)

        if request_state.top_logprobs > 0:
            request_state.top_logprobs_buffer.append(
                summarize_top_logprobs(
                    self._tokenizer, response.logprobs, request_state.top_logprobs
                )
            )

        if request_state.stop_string_processor is not None:
            stop_result = request_state.stop_string_processor.process_token(token_id)
            if stop_result.status == "full_stop":
                stop_condition = self._handle_stop_string_detected(
                    request_state, stop_result
                )
                prompt_cache = self._extract_prompt_cache_for_request_slot(request_slot_id)
                self._batch_generator.remove([request_slot_id])
                result = self._build_result(
                    request_slot_id, request_state, stop_condition
                )
                self._finish_request(request_slot_id, request_state, prompt_cache)
                return result
            if stop_result.status in ("partial_match", "multi_byte"):
                return None

        if response.finish_reason is not None:
            request_state.detokenizer.finalize()
            if request_state.detokenizer.last_segment != "":
                request_state.text_buffer += request_state.detokenizer.last_segment

            stop_condition = self._finish_reason_to_stop_condition(
                response.finish_reason, token_id
            )
            prompt_cache = response.prompt_cache
            result = self._build_result(request_slot_id, request_state, stop_condition)
            self._finish_request(request_slot_id, request_state, prompt_cache)
            return result

        if request_state.text_buffer != "":
            return self._build_result(request_slot_id, request_state, None)
        return None

    def _handle_stop_string_detected(
        self, request_state: BatchRequestState, stop_result: StopStringProcessorResult
    ) -> GenerationStopCondition:
        request_state.detokenizer.finalize()
        if request_state.detokenizer.last_segment != "":
            request_state.text_buffer += request_state.detokenizer.last_segment

        stop_string = stop_result.stop_string or ""
        stop_string_start = request_state.text_buffer.find(stop_string)
        if stop_string_start != -1:
            request_state.text_buffer = request_state.text_buffer[:stop_string_start]
        stop_tokens = stop_result.stop_tokens if stop_result.stop_tokens is not None else []
        return GenerationStopCondition(
            stop_reason="stop_string",
            stop_string=stop_string,
            stop_tokens=stop_tokens,
        )

    def _finish_reason_to_stop_condition(
        self, finish_reason: str, token_id: int
    ) -> GenerationStopCondition:
        if finish_reason == "stop":
            return GenerationStopCondition(
                stop_reason="eos_token",
                stop_string=self._tokenizer.decode(token_id),
                stop_tokens=[token_id],
            )
        if finish_reason == "length":
            return GenerationStopCondition(
                stop_reason="length",
                stop_string="",
                stop_tokens=[token_id],
            )
        return GenerationStopCondition(
            stop_reason="length",
            stop_string="",
            stop_tokens=[token_id],
        )

    def _build_result(
        self,
        request_slot_id: int,
        request_state: BatchRequestState,
        stop_condition: Optional[GenerationStopCondition],
    ) -> BatchGenerationResult:
        tokens = list(request_state.token_buffer)
        top_logprobs = list(request_state.top_logprobs_buffer)
        text = request_state.text_buffer
        request_state.token_buffer = []
        request_state.top_logprobs_buffer = []
        request_state.text_buffer = ""
        return BatchGenerationResult(
            request_slot_id=request_slot_id,
            text=text,
            tokens=tokens,
            top_logprobs=top_logprobs,
            stop_condition=stop_condition,
        )

    def _extract_prompt_cache_for_request_slot(
        self, request_slot_id: int
    ) -> Optional[List[Any]]:
        batch = self._batch_generator.active_batch
        if batch is None:
            return None
        if request_slot_id not in batch.uids:
            return None
        request_index = batch.uids.index(request_slot_id)
        return batch.extract_cache(request_index)

    def _finish_request(
        self,
        request_slot_id: int,
        request_state: BatchRequestState,
        prompt_cache: Optional[List[Any]],
    ) -> None:
        if request_state.should_cache and prompt_cache is not None:
            self._prompt_cache_store.insert_cache(
                request_state.session_id, request_state.cache_key, prompt_cache
            )
        if request_slot_id in self._request_states:
            del self._request_states[request_slot_id]


def create_batch_generator(
    model_kit: ModelKit | VisionModelKit,
    *,
    completion_batch_size: int = 32,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[
        Callable[[List[Tuple[int, int, int]]], None]
    ] = None,
) -> BatchGeneratorWrapper:
    """
    Create a batch generator wrapper for continuous batching inference.

    This function initializes a BatchGeneratorWrapper with batching-specific settings
    and an optional prompt progress callback for reporting prefill progress.

    Args:
        model_kit (ModelKit | VisionModelKit): The model to use for batching.
        completion_batch_size (int): Maximum number of concurrent completion slots.
        prefill_batch_size (int): Maximum number of prompts prefilling concurrently.
        prefill_step_size (int): Number of prompt tokens to prefill per step.
        prompt_progress_callback (Optional[Callable[[List[Tuple[int, int, int]]], None]]):
            Callback that receives prompt prefill progress events.

    Returns:
        BatchGeneratorWrapper: A wrapper that manages batch request insertion and iteration.
    """
    return BatchGeneratorWrapper(
        model_kit,
        completion_batch_size=completion_batch_size,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
        prompt_progress_callback=prompt_progress_callback,
    )