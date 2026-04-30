"""Local mlx-vlm batcher shim with history restore support.

The public insert/response shape intentionally mirrors the text batcher so
`BatchedVisionModelKit` can re-enter generation with restored history. In this
V1 implementation, the only extra decode-side payload we carry is RoPE deltas.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
)
from mlx_vlm.generate import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    _left_pad_prompts,
    _make_cache,
    wired_limit,
)

# The local batcher owns generation on the mlx-engine generation thread, so it
# should not mutate mlx-vlm's module-global generation stream.
_generation_stream = mx.new_thread_local_stream(mx.default_device())

PromptCacheSaveCallback = Callable[[list[Any], list[int]], None]


@contextlib.contextmanager
def use_generation_stream():
    """Run MLX model work on the local batcher's generation stream."""
    with mx.stream(_generation_stream):
        yield


@dataclass
class _PendingSequence:
    uid: int
    prompt: list[int]
    max_tokens: int
    sampler: Optional[Callable[[mx.array], mx.array]]
    cache: Optional[list[Any]]
    all_tokens: list[int]
    rope_deltas: Any | None
    prompt_kwargs: dict
    cache_save_points: list[int]
    prompt_cache_save_callback: Optional[PromptCacheSaveCallback]


def _append_next_periodic_save_point(save_points: list[int], save_point: int) -> None:
    if not save_points:
        save_points.append(save_point + DEFAULT_PREFIX_CHUNK_SIZE)


def _merge_caches(caches: list[list[Any]]) -> list[Any]:
    if not caches:
        return []

    batch_cache = []
    for i in range(len(caches[0])):
        if not hasattr(caches[0][i], "merge"):
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
        batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _normalize_rope_deltas(rope_deltas: Any) -> Any:
    if rope_deltas is None:
        return None
    if rope_deltas.ndim == 0:
        return rope_deltas.reshape(1, 1)
    if rope_deltas.ndim == 1:
        return rope_deltas[:, None]
    return rope_deltas


def _empty_rope_deltas_like(rope_deltas: Any, count: int) -> Any:
    return mx.zeros((count, *rope_deltas.shape[1:]), dtype=rope_deltas.dtype)


def _extend_optional_rope_deltas(
    rope_deltas: Any | None,
    count: int,
    other_rope_deltas: Any | None,
    other_count: int,
) -> Any | None:
    if rope_deltas is None and other_rope_deltas is None:
        return None
    # Mixed Qwen/text batches need a per-row value; no side state means zero.
    if rope_deltas is None:
        rope_deltas = _empty_rope_deltas_like(other_rope_deltas, count)
    if other_rope_deltas is None:
        other_rope_deltas = _empty_rope_deltas_like(rope_deltas, other_count)
    return mx.concatenate([rope_deltas, other_rope_deltas])


def _sample_with_per_request_samplers(
    logprobs: mx.array,
    samplers: list[Optional[Callable[[mx.array], mx.array]]],
    fallback_sampler: Callable[[mx.array], mx.array],
) -> mx.array:
    if any(samplers):
        return mx.concatenate(
            [
                (sampler or fallback_sampler)(logprobs[i : i + 1])
                for i, sampler in enumerate(samplers)
            ],
            axis=0,
        )
    return fallback_sampler(logprobs)


class GenerationBatch:
    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[list[tuple[int, float]]] = None
        prompt_cache: Optional[list[Any]] = None
        all_tokens: Optional[list[int]] = None
        rope_deltas: Any | None = None

    def __init__(
        self,
        model: nn.Module,
        uids: list[int],
        inputs: mx.array,
        prompt_cache: list[Any],
        sampler: Callable[[mx.array], mx.array],
        samplers: Optional[list[Optional[Callable[[mx.array], mx.array]]]],
        stop_criteria,
        max_tokens: list[int],
        top_logprobs_k: int = 0,
        all_tokens: Optional[list[list[int]]] = None,
        rope_deltas: Any | None = None,
        cache_save_points: Optional[list[list[int]]] = None,
        prompt_cache_save_callback: Optional[PromptCacheSaveCallback] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.samplers = samplers if samplers is not None else [None] * len(self.uids)
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.top_logprobs_k = top_logprobs_k

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self._next_lps = None
        self._next_top_idx = None
        self._next_top_lp = None
        self._rope_deltas = None
        self.tokens = (
            [list(tokens) for tokens in all_tokens]
            if all_tokens is not None
            else [[] for _ in uids]
        )
        self._cache_save_points = (
            [list(save_points) for save_points in cache_save_points]
            if cache_save_points is not None
            else [[] for _ in uids]
        )
        self._cache_save_callbacks = [prompt_cache_save_callback for _ in uids]
        self._rope_deltas = _normalize_rope_deltas(rope_deltas)

    def __len__(self):
        return len(self.uids)

    def _step(self):
        self._current_tokens = self._next_tokens
        self._current_lps = self._next_lps
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = self._rope_deltas

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = _sample_with_per_request_samplers(
            logprobs,
            self.samplers,
            self.sampler,
        )

        self._next_tokens = sampled
        prev_top_idx = self._next_top_idx
        prev_top_lp = self._next_top_lp

        eval_targets = [self._next_tokens]
        self._next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
        eval_targets.append(self._next_lps)

        k = self.top_logprobs_k
        if k > 0:
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self._next_top_idx = top_idx
            self._next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self._next_top_idx = None
            self._next_top_lp = None

        mx.async_eval(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            mx.eval(*to_eval)
            tokens = inputs.tolist()
            lp_list = self._current_lps.tolist()
            top_idx_list = prev_top_idx.tolist() if prev_top_idx is not None else None
            top_lp_list = prev_top_lp.tolist() if prev_top_lp is not None else None
        else:
            mx.eval(inputs)
            tokens = inputs.tolist()
            lp_list = None
            top_idx_list = None
            top_lp_list = None

        for seq_tokens, token in zip(self.tokens, tokens):
            seq_tokens.append(token)
        return tokens, lp_list, top_idx_list, top_lp_list

    def extract_cache(self, idx: int) -> list[Any]:
        return [cache.extract(idx) for cache in self.prompt_cache]

    def extract_rope_deltas(self, idx: int) -> Any | None:
        if self._rope_deltas is None:
            return None
        return mx.contiguous(self._rope_deltas[idx : idx + 1])

    def append_prefilled_sequence(self, prefilled: "GenerationBatch"):
        """Append the one sequence that just finished prompt prefill."""
        count = len(self.uids)
        prefilled_count = len(prefilled.uids)
        self.uids.extend(prefilled.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, prefilled.prompt_cache)
        self.samplers.extend(prefilled.samplers)
        self.max_tokens.extend(prefilled.max_tokens)
        self._num_tokens.extend(prefilled._num_tokens)

        if self._current_tokens is None:
            self._current_tokens = prefilled._current_tokens
            self._current_lps = prefilled._current_lps
        elif prefilled._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, prefilled._current_tokens]
            )
            if self._current_lps is not None and prefilled._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, prefilled._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = prefilled._next_tokens
            self._next_lps = prefilled._next_lps
            self._next_top_idx = prefilled._next_top_idx
            self._next_top_lp = prefilled._next_top_lp
        elif prefilled._next_tokens is not None:
            self._next_tokens = mx.concatenate(
                [self._next_tokens, prefilled._next_tokens]
            )
            if self._next_lps is not None and prefilled._next_lps is not None:
                self._next_lps = mx.concatenate([self._next_lps, prefilled._next_lps])

            if (
                self._next_top_idx is not None
                and prefilled._next_top_idx is not None
                and self._next_top_idx.shape[-1] == prefilled._next_top_idx.shape[-1]
            ):
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, prefilled._next_top_idx]
                )
                self._next_top_lp = mx.concatenate(
                    [self._next_top_lp, prefilled._next_top_lp]
                )
            else:
                self._next_top_idx = None
                self._next_top_lp = None

        self._rope_deltas = _extend_optional_rope_deltas(
            self._rope_deltas,
            count,
            prefilled._rope_deltas,
            prefilled_count,
        )

        self.tokens.extend(prefilled.tokens)
        self._cache_save_points.extend(prefilled._cache_save_points)
        self._cache_save_callbacks.extend(prefilled._cache_save_callbacks)

    def filter(self, keep: list[int]):
        self.uids = [self.uids[idx] for idx in keep]
        self.samplers = [self.samplers[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self.tokens = [self.tokens[idx] for idx in keep]
        self._cache_save_points = [self._cache_save_points[idx] for idx in keep]
        self._cache_save_callbacks = [self._cache_save_callbacks[idx] for idx in keep]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            self._rope_deltas = None
            return

        keep_arr = mx.array(keep, mx.int32)
        for cache in self.prompt_cache:
            cache.filter(keep_arr)
        if self._next_tokens is not None:
            self._next_tokens = self._next_tokens[keep_arr]
        if self._next_lps is not None:
            self._next_lps = self._next_lps[keep_arr]
        if self._next_top_idx is not None:
            self._next_top_idx = self._next_top_idx[keep_arr]
            self._next_top_lp = self._next_top_lp[keep_arr]
        if self._rope_deltas is not None:
            self._rope_deltas = self._rope_deltas[keep_arr]

    def _emit_cache_save_snapshot(self, idx: int) -> None:
        callback = self._cache_save_callbacks[idx]
        if callback is None:
            return

        current_len = len(self.tokens[idx])
        save_points = self._cache_save_points[idx]
        while save_points and save_points[0] <= current_len:
            save_point = save_points.pop(0)
            if save_point == current_len:
                callback(self.extract_cache(idx), list(self.tokens[idx]))
                _append_next_periodic_save_point(save_points, save_point)

    def next(self) -> list[Response]:
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            self._emit_cache_save_snapshot(i)

            top_lp = None
            if top_idx_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                    prompt_cache=self.extract_cache(i) if finish_reason else None,
                    all_tokens=list(self.tokens[i]) if finish_reason else None,
                    rope_deltas=self.extract_rope_deltas(i) if finish_reason else None,
                )
            )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(cls, model, sampler, stop_criteria, top_logprobs_k=0):
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.samplers = []
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.top_logprobs_k = top_logprobs_k
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        batch.tokens = []
        batch._cache_save_points = []
        batch._cache_save_callbacks = []
        return batch


class PromptProcessingBatch:
    """One request being prefetched before it joins the decode batch."""

    @dataclass
    class Response:
        uid: int
        progress: tuple[int, int]
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uid: int,
        input_ids: list[int],
        max_tokens: int,
        sampler: Optional[Callable[[mx.array], mx.array]],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        cache: Optional[list[Any]] = None,
        all_tokens: Optional[list[int]] = None,
        rope_deltas: Any | None = None,
        cache_save_points: Optional[list[int]] = None,
        prompt_cache_save_callback: Optional[PromptCacheSaveCallback] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = [uid]
        self.uid = uid
        self.max_tokens = [max_tokens]
        self.samplers = [sampler]
        self.prefill_step_size = prefill_step_size

        self._input_ids = _left_pad_prompts([input_ids], max_length=len(input_ids))
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs
        self._prompt_token_ids = list(input_ids)
        self._all_tokens = list(all_tokens) if all_tokens is not None else []
        self._processed_prefix_len = len(self._all_tokens)
        self._cache_save_points = list(cache_save_points or [])
        self._prompt_cache_save_callback = prompt_cache_save_callback
        self._rope_deltas = _normalize_rope_deltas(rope_deltas)

        if cache is None:
            self.prompt_cache = _make_cache(
                model,
                [0],
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
            )
        else:
            self.prompt_cache = _merge_caches([cache])

    def __len__(self):
        return len(self.uids)

    def progress_responses(self, *, end_of_prompt: bool) -> list[Response]:
        total = len(self._all_tokens) + len(self._prompt_token_ids)
        return [
            self.Response(
                self.uid,
                (self._processed_prefix_len, total),
                end_of_prompt,
                end_of_prompt,
            )
        ]

    def needs_processing(self):
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return False

        remaining_tokens = self._inputs_embeds.shape[1]
        if remaining_tokens <= 1:
            return False

        next_save_point = self._next_cache_save_point()
        if (
            next_save_point is not None
            and next_save_point < self._processed_prefix_len + remaining_tokens
        ):
            return True

        return remaining_tokens > self.prefill_step_size

    def prompt_step(self) -> int:
        if not self.needs_processing():
            return 0

        remaining_tokens = self._inputs_embeds.shape[1]
        n = min(self.prefill_step_size, remaining_tokens - 1)
        next_save_point = self._next_cache_save_point()
        if next_save_point is not None:
            save_point_delta = next_save_point - self._processed_prefix_len
            if 0 < save_point_delta < remaining_tokens:
                n = min(n, save_point_delta)

        self._language_model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **self._prompt_kwargs,
        )
        mx.eval([c.state for c in self.prompt_cache])
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        self._processed_prefix_len += n
        self._emit_cache_save_snapshots()
        mx.clear_cache()
        return n

    def _next_cache_save_point(self) -> Optional[int]:
        processed = self._processed_prefix_len
        for save_point in self._cache_save_points:
            if save_point > processed:
                return save_point
        return None

    def _emit_cache_save_snapshots(self) -> None:
        if self._prompt_cache_save_callback is None:
            return

        processed = self._processed_prefix_len
        save_points = self._cache_save_points
        while save_points and save_points[0] <= processed:
            save_point = save_points.pop(0)
            if save_point == processed:
                suffix_len = processed - len(self._all_tokens)
                snapshot_tokens = self._all_tokens + self._prompt_token_ids[:suffix_len]
                self._prompt_cache_save_callback(
                    [
                        cache.extract(0) if hasattr(cache, "extract") else cache
                        for cache in self.prompt_cache
                    ],
                    snapshot_tokens,
                )
                _append_next_periodic_save_point(save_points, save_point)

    def generate(
        self, sampler, stop_criteria, top_logprobs_k=0
    ) -> tuple[GenerationBatch, list[Response]]:
        output = self._language_model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **self._prompt_kwargs,
        )
        self._processed_prefix_len = len(self._all_tokens) + len(self._prompt_token_ids)
        self._emit_cache_save_snapshots()
        prompt_responses = self.progress_responses(end_of_prompt=True)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = _sample_with_per_request_samplers(
            logprobs,
            self.samplers,
            sampler,
        )

        gen_batch = GenerationBatch(
            model=self.model,
            uids=list(self.uids),
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            sampler=sampler,
            samplers=list(self.samplers),
            stop_criteria=stop_criteria,
            max_tokens=list(self.max_tokens),
            top_logprobs_k=top_logprobs_k,
            all_tokens=[self._all_tokens + self._prompt_token_ids],
            rope_deltas=self._rope_deltas,
            cache_save_points=[list(self._cache_save_points)],
            prompt_cache_save_callback=self._prompt_cache_save_callback,
        )
        gen_batch._next_lps = logprobs[mx.arange(first_tokens.shape[0]), first_tokens]

        if top_logprobs_k > 0:
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch._next_top_idx = top_idx
            gen_batch._next_top_lp = top_lp

        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = _normalize_rope_deltas(
            getattr(language_model, "_rope_deltas", None)
        )
        if rope_deltas is not None:
            gen_batch._rope_deltas = rope_deltas

        self.uids = []
        self.prompt_cache = []
        return gen_batch, prompt_responses


class BatchGenerator:
    """MLX-VLM-style continuous batcher with single-request prompt prefill."""

    def __init__(
        self,
        model,
        stop_criteria,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        top_logprobs_k: int = 0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.top_logprobs_k = top_logprobs_k
        self.stop_criteria = stop_criteria
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.completion_batch_size = completion_batch_size

        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences: list[_PendingSequence] = []

        self._steps_counter = 0

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [_generation_stream]))
        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.stop_criteria,
            top_logprobs_k=self.top_logprobs_k,
        )

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompts,
        max_tokens: Union[list[int], int, None] = None,
        caches: Optional[list[Optional[list[Any]]]] = None,
        all_tokens: Optional[list[list[int]]] = None,
        rope_deltas: Optional[list[Any | None]] = None,
        samplers: Optional[list[Optional[Callable[[mx.array], mx.array]]]] = None,
        prompt_kwargs: Optional[list[dict]] = None,
        cache_save_points: Optional[list[list[int]]] = None,
        prompt_cache_save_callback: Optional[PromptCacheSaveCallback] = None,
    ):
        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if caches is None:
            caches = [None] * len(prompts)
        if all_tokens is None:
            all_tokens = [[] for _ in prompts]
        if rope_deltas is None:
            rope_deltas = [None] * len(prompts)
        if samplers is None:
            samplers = [None] * len(prompts)
        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if cache_save_points is None:
            cache_save_points = [[] for _ in prompts]

        uids = []
        for p, m, c, at, rd, s, kw, save_points in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            rope_deltas,
            samplers,
            prompt_kwargs,
            cache_save_points,
        ):
            self._unprocessed_sequences.append(
                _PendingSequence(
                    uid=self.uid_count,
                    prompt=p,
                    max_tokens=m,
                    sampler=s,
                    cache=c,
                    all_tokens=at,
                    rope_deltas=rd,
                    prompt_kwargs=kw,
                    cache_save_points=save_points,
                    prompt_cache_save_callback=prompt_cache_save_callback,
                )
            )
            uids.append(self.uid_count)
            self.uid_count += 1

        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda sequence: len(sequence.prompt)
        )
        return uids

    def next(self):
        with mx.stream(_generation_stream):
            return self._next()

    def remove(self, uid) -> bool:
        with mx.stream(_generation_stream):
            for i, sequence in enumerate(self._unprocessed_sequences):
                if sequence.uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    def _next(self):
        generation_responses = []
        prompt_responses = []

        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                self._prompt_batch.prompt_step()
                prompt_responses = self._prompt_batch.progress_responses(
                    end_of_prompt=False
                )
                return prompt_responses, generation_responses

            gen_batch, prompt_responses = self._prompt_batch.generate(
                self.sampler,
                self.stop_criteria,
                top_logprobs_k=self.top_logprobs_k,
            )
            self._generation_batch.append_prefilled_sequence(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= 1:
            sequence = self._unprocessed_sequences.pop(0)

            inputs_embeds = sequence.prompt_kwargs.get("inputs_embeds")
            prompt_kwargs = {
                key: value
                for key, value in sequence.prompt_kwargs.items()
                if key != "inputs_embeds"
            }

            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required")

            self._prompt_batch = PromptProcessingBatch(
                model=self.model,
                uid=sequence.uid,
                input_ids=sequence.prompt,
                max_tokens=sequence.max_tokens,
                sampler=sequence.sampler,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=prompt_kwargs,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                cache=sequence.cache,
                all_tokens=sequence.all_tokens,
                rope_deltas=sequence.rope_deltas,
                cache_save_points=sequence.cache_save_points,
                prompt_cache_save_callback=sequence.prompt_cache_save_callback,
            )

            if self._prompt_batch.needs_processing():
                self._prompt_batch.prompt_step()
                prompt_responses = self._prompt_batch.progress_responses(
                    end_of_prompt=False
                )
            else:
                gen_batch, prompt_responses = self._prompt_batch.generate(
                    self.sampler,
                    self.stop_criteria,
                    top_logprobs_k=self.top_logprobs_k,
                )
                self._generation_batch.append_prefilled_sequence(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses
