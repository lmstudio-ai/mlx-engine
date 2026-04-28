"""Local mlx-vlm batcher shim with history restore support.

The public insert/response shape intentionally mirrors the text batcher so
`BatchedVisionModelKit` can re-enter generation with restored history. In this
V1 implementation, the only extra decode-side payload we carry is
`{"rope_deltas": ...}`.
"""

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.generate as mlx_vlm_generate
from mlx_vlm.generate import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerator as MlxVlmBatchGenerator,
    GenerationBatch as MlxVlmGenerationBatch,
    PromptProcessingBatch as MlxVlmPromptProcessingBatch,
    _left_pad_prompts,
    _make_cache,
    cache as vlm_cache,
    wired_limit,
)

# MLX 0.31.2 keeps regular GPU streams thread-affine. Our batched vision backend
# owns generation on a worker thread, so force mlx-vlm to share a thread-local
# generation stream object instead of the upstream module-global Stream.
generation_stream = mlx_vlm_generate.generation_stream = mx.new_thread_local_stream(
    mx.default_device()
)


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


def _normalize_rope_deltas(rope_deltas: Any) -> Any:
    if rope_deltas is None:
        return None
    if rope_deltas.ndim == 0:
        return rope_deltas.reshape(1, 1)
    if rope_deltas.ndim == 1:
        return rope_deltas[:, None]
    return rope_deltas


def _merge_rope_deltas_from_decode_states(
    decode_states: list[Optional[dict[str, Any]]],
) -> Any:
    # V1 keeps prefill_batch_size at 1, so all-or-none restored decode state is
    # good enough until the disk cache actually starts feeding this path.
    if not decode_states or any(state is None for state in decode_states):
        return None

    rope_deltas = [
        _normalize_rope_deltas(state.get("rope_deltas")) for state in decode_states
    ]
    if any(delta is None for delta in rope_deltas):
        return None

    if len(rope_deltas) == 1:
        return rope_deltas[0]
    return mx.concatenate(rope_deltas)


class GenerationBatch(MlxVlmGenerationBatch):
    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[list[tuple[int, float]]] = None
        prompt_cache: Optional[list[Any]] = None
        all_tokens: Optional[list[int]] = None
        decode_state: Optional[dict[str, Any]] = None

    def __init__(
        self,
        model: nn.Module,
        uids: list[int],
        inputs: mx.array,
        prompt_cache: list[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: list[int],
        top_logprobs_k: int = 0,
        all_tokens: Optional[list[list[int]]] = None,
        decode_state: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            model=model,
            uids=uids,
            inputs=inputs,
            prompt_cache=prompt_cache,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=max_tokens,
            top_logprobs_k=top_logprobs_k,
        )
        self.tokens = (
            [list(tokens) for tokens in all_tokens]
            if all_tokens is not None
            else [[] for _ in uids]
        )
        if decode_state is not None:
            self._rope_deltas = _normalize_rope_deltas(decode_state.get("rope_deltas"))

    def _step(self):
        with mx.stream(generation_stream):
            tokens, lp_list, top_idx_list, top_lp_list = super()._step()
        for seq_tokens, token in zip(self.tokens, tokens):
            seq_tokens.append(token)
        return tokens, lp_list, top_idx_list, top_lp_list

    def extract_cache(self, idx: int) -> list[Any]:
        return [cache.extract(idx) for cache in self.prompt_cache]

    def extract_decode_state(self, idx: int) -> Optional[dict[str, Any]]:
        if self._rope_deltas is None:
            return None
        # The local batcher still exposes decode_state for compatibility, but
        # V1 only needs rope_deltas here.
        return {
            "rope_deltas": mx.contiguous(self._rope_deltas[idx : idx + 1]),
        }

    def extend(self, other: "GenerationBatch"):
        super().extend(other)
        self.tokens.extend(other.tokens)

    def filter(self, keep: list[int]):
        super().filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]

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
                    decode_state=(
                        self.extract_decode_state(i) if finish_reason else None
                    ),
                )
            )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls, model, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ):
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        batch.tokens = []
        return batch


class PromptProcessingBatch(MlxVlmPromptProcessingBatch):
    def __init__(
        self,
        model: nn.Module,
        uids: list[int],
        input_ids: list[list[int]],
        max_tokens: list[int],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        caches: Optional[list[Optional[list[Any]]]] = None,
        all_tokens: Optional[list[list[int]]] = None,
        decode_states: Optional[list[Optional[dict[str, Any]]]] = None,
        cache_boundaries: Optional[list[list[int]]] = None,
        prompt_cache_boundary_callback: Optional[
            Callable[[int, int, list[Any], Optional[dict[str, Any]]], None]
        ] = None,
    ):
        self.model = model
        self.uids = uids
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        left_padding = [max_length - length for length in lengths]
        self._total_prompt_tokens = sum(lengths)

        self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs
        self._prompt_token_ids = [list(ids) for ids in input_ids]
        self._all_tokens = (
            [list(tokens) for tokens in all_tokens]
            if all_tokens is not None
            else [[] for _ in uids]
        )
        self._processed_prefix_lengths = [len(tokens) for tokens in self._all_tokens]
        self._cache_boundaries = (
            [list(boundaries) for boundaries in cache_boundaries]
            if cache_boundaries is not None
            else [[] for _ in uids]
        )
        self._prompt_cache_boundary_callback = prompt_cache_boundary_callback
        self._rope_deltas = (
            _merge_rope_deltas_from_decode_states(decode_states)
            if decode_states is not None
            else None
        )

        if caches is None or all(seq_cache is None for seq_cache in caches):
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
            )
        else:
            prepared_caches = [
                seq_cache
                if seq_cache is not None
                else vlm_cache.make_prompt_cache(model)
                for seq_cache in caches
            ]
            self.prompt_cache = _merge_caches(prepared_caches)

    def needs_processing(self):
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return False

        remaining_tokens = self._inputs_embeds.shape[1]
        if remaining_tokens <= 1:
            return False

        next_boundary = self._next_cache_boundary()
        if (
            next_boundary is not None
            and next_boundary < self._processed_prefix_lengths[0] + remaining_tokens
        ):
            return True

        return remaining_tokens > self.prefill_step_size

    def prompt_step(self) -> int:
        if not self.needs_processing():
            return 0

        remaining_tokens = self._inputs_embeds.shape[1]
        n = min(self.prefill_step_size, remaining_tokens - 1)
        next_boundary = self._next_cache_boundary()
        if next_boundary is not None:
            boundary_delta = next_boundary - self._processed_prefix_lengths[0]
            if 0 < boundary_delta < remaining_tokens:
                n = min(n, boundary_delta)

        self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **self._prompt_kwargs,
        )
        mx.eval([c.state for c in self.prompt_cache])
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        self._processed_prefix_lengths = [
            processed + n for processed in self._processed_prefix_lengths
        ]
        self._emit_boundary_snapshots()
        mx.clear_cache()
        return n

    def _next_cache_boundary(self) -> Optional[int]:
        if len(self.uids) != 1 or not self._cache_boundaries:
            return None

        processed = self._processed_prefix_lengths[0]
        for boundary in self._cache_boundaries[0]:
            if boundary > processed:
                return boundary
        return None

    def _current_decode_state(self) -> Optional[dict[str, Any]]:
        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = _normalize_rope_deltas(
            getattr(language_model, "_rope_deltas", None)
        )
        if rope_deltas is None:
            return None
        return {"rope_deltas": rope_deltas}

    def _emit_boundary_snapshots(self) -> None:
        if self._prompt_cache_boundary_callback is None or len(self.uids) != 1:
            return

        processed = self._processed_prefix_lengths[0]
        boundaries = self._cache_boundaries[0]
        while boundaries and boundaries[0] <= processed:
            boundary = boundaries.pop(0)
            if boundary == processed:
                self._prompt_cache_boundary_callback(
                    self.uids[0],
                    boundary,
                    self.prompt_cache,
                    self._current_decode_state(),
                )

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **self._prompt_kwargs,
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = sampler(logprobs)

        gen_batch = GenerationBatch(
            model=self.model,
            uids=list(self.uids),
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=list(self.max_tokens),
            top_logprobs_k=top_logprobs_k,
            all_tokens=[
                prefix + prompt_tokens
                for prefix, prompt_tokens in zip(
                    self._all_tokens, self._prompt_token_ids
                )
            ],
            decode_state=(
                {"rope_deltas": self._rope_deltas}
                if self._rope_deltas is not None
                else None
            ),
        )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs:
            gen_batch._next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

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
        return gen_batch


class BatchGenerator(MlxVlmBatchGenerator):
    def __init__(
        self,
        model,
        processor,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        prompt_cache=None,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [generation_stream]))
        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
        )

    def insert(
        self,
        prompts,
        max_tokens: Union[list[int], int, None] = None,
        caches: Optional[list[Optional[list[Any]]]] = None,
        all_tokens: Optional[list[list[int]]] = None,
        decode_states: Optional[list[Optional[dict[str, Any]]]] = None,
        prompt_kwargs: Optional[list[dict]] = None,
        cache_boundaries: Optional[list[list[int]]] = None,
        prompt_cache_boundary_callback: Optional[
            Callable[[int, int, list[Any], Optional[dict[str, Any]]], None]
        ] = None,
    ):
        # Keep the mlx-lm-style decode_states interface, but the only supported
        # payload today is {"rope_deltas": ...}.
        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if caches is None:
            caches = [None] * len(prompts)
        if all_tokens is None:
            all_tokens = [[] for _ in prompts]
        if decode_states is None:
            decode_states = [None] * len(prompts)
        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if cache_boundaries is None:
            cache_boundaries = [[] for _ in prompts]

        uids = []
        for p, m, c, at, ds, kw, cb in zip(
            prompts,
            max_tokens,
            caches,
            all_tokens,
            decode_states,
            prompt_kwargs,
            cache_boundaries,
        ):
            self._unprocessed_sequences.append(
                (
                    self.uid_count,
                    p,
                    m,
                    c,
                    at,
                    ds,
                    kw,
                    cb,
                    prompt_cache_boundary_callback,
                )
            )
            uids.append(self.uid_count)
            self.uid_count += 1

        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def next(self, **kwargs):
        with mx.stream(generation_stream):
            return self._next(**kwargs)

    def remove(self, uid) -> bool:
        with mx.stream(generation_stream):
            for i, (seq_uid, _, _, _, _, _, _, _, _) in enumerate(
                self._unprocessed_sequences
            ):
                if seq_uid == uid:
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

    def _next(self, **kwargs):
        generation_responses = []
        prompt_responses = []

        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
                self._prompt_tokens_counter += n
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            self._prompt_time_counter += time.perf_counter() - tic
            self._generation_batch.extend(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            caches_list = [s[3] for s in sequences]
            all_tokens_list = [s[4] for s in sequences]
            decode_states_list = [s[5] for s in sequences]
            prompt_kwargs_list = [s[6] for s in sequences]
            cache_boundaries_list = [s[7] for s in sequences]
            prompt_cache_boundary_callback = sequences[0][8]

            inputs_embeds = None
            merged_kwargs = {}
            for kw in prompt_kwargs_list:
                if kw:
                    inputs_embeds = kw.get("inputs_embeds", inputs_embeds)
                    merged_kwargs = {
                        key: value
                        for key, value in kw.items()
                        if key != "inputs_embeds"
                    }
                    break

            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required")

            batch_size = len(uids)
            for key, value in merged_kwargs.items():
                if isinstance(value, mx.array) and value.ndim > 0:
                    merged_kwargs[key] = value[:batch_size]

            self._prompt_batch = PromptProcessingBatch(
                model=self.model,
                uids=uids,
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                caches=caches_list,
                all_tokens=all_tokens_list,
                decode_states=decode_states_list,
                cache_boundaries=cache_boundaries_list,
                prompt_cache_boundary_callback=prompt_cache_boundary_callback,
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                self._prompt_time_counter += time.perf_counter() - tic
                self._generation_batch.extend(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses
