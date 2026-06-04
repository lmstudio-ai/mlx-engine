"""Local mlx-vlm-style batcher with history restore support.

`BatchedVisionModelKit` owns prompt/image prep and passes ready embeddings here.
The batcher only owns language-model prefill/decode plus RoPE delta handoff.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    extend_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    DEFAULT_PREFIX_CHUNK_SIZE,
    PromptImageSpan,
    PromptPrefixChunk,
)
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    drop_prompt_kwargs_prefix,
    slice_prompt_kwargs,
)
from mlx_engine.model_kit.patches.gemma4 import (
    prepare_cached_suffix_prompt_kwargs as prepare_gemma4_cached_suffix_prompt_kwargs,
    visual_prefill_prefix_len as gemma4_visual_prefill_prefix_len,
)
from mlx_vlm.generate import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    _left_pad_prompts,
    wired_limit,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)

logger = logging.getLogger(__name__)

PromptCacheSaveCallback = Callable[
    [list[Any], list[PromptPrefixChunk], int, int, int], None
]
LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


@dataclass
class _PrefixCacheSaveState:
    chunks: list[PromptPrefixChunk]
    next_chunk_idx: int
    image_spans: list[PromptImageSpan]
    callback: PromptCacheSaveCallback | None


@dataclass
class _GenerationRow:
    """Python-side state for one row in a decode batch."""

    uid: int
    sampler: Callable[[mx.array], mx.array]
    logits_processors: list[LogitsProcessor]
    max_tokens: int
    top_logprobs: int
    tokens: list[int]
    prefix_cache_save_state: _PrefixCacheSaveState
    num_tokens: int = 0


@dataclass
class _PendingSequence:
    uid: int
    prompt: list[int]
    max_tokens: int
    top_logprobs: int
    sampler: Callable[[mx.array], mx.array]
    logits_processors: list[LogitsProcessor]
    inputs_embeds: mx.array
    cache: Optional[list[Any]]
    all_tokens: list[int]
    rope_deltas: Any | None
    prompt_kwargs: dict
    prefix_cache_save_state: _PrefixCacheSaveState


def _batch_single_cache(cache: list[Any]) -> list[Any]:
    """Convert one scalar restored cache into a batch-size-one cache."""
    batch_cache = []
    for layer_cache in cache:
        if not hasattr(layer_cache, "merge"):
            raise ValueError(
                f"{type(layer_cache)} does not yet support batching with history"
            )
        batch_cache.append(layer_cache.merge([layer_cache]))
    return batch_cache


def _is_scalar_prompt_cache(prompt_cache: list[Any]) -> bool:
    """Return True for ordinary one-request caches, not batch-aware caches."""
    return any(not hasattr(cache, "filter") for cache in prompt_cache)


def _ensure_batch_cache(prompt_cache: list[Any]) -> list[Any]:
    if _is_scalar_prompt_cache(prompt_cache):
        return _batch_single_cache(prompt_cache)
    return prompt_cache


def _sync_scalar_rope_deltas(model: nn.Module, prompt_cache: list[Any], rope_deltas):
    if not _is_scalar_prompt_cache(prompt_cache) or rope_deltas is None:
        return
    language_model = getattr(model, "language_model", model)
    if hasattr(language_model, "_rope_deltas"):
        # Some Qwen scalar decode paths ignore the kwarg and read this side state.
        language_model._rope_deltas = rope_deltas


def _clear_qwen3_5_text_rope_state(model: nn.Module, prompt_kwargs: dict) -> None:
    """Clear stale Qwen MRoPE side state before text-only prompt prefill.

    Qwen3.5-family VLM decode stores RoPE deltas on the shared language model.
    Continuous batching can run an active image decode step immediately before a
    new text-only prefill, so text-only prompt calls must explicitly enter the
    model with no image/MRoPE state.
    """
    if "position_ids" in prompt_kwargs or "rope_deltas" in prompt_kwargs:
        return
    language_model = getattr(model, "language_model", model)
    if not str(getattr(language_model, "model_type", "")).startswith("qwen3_5"):
        return
    language_model._position_ids = None
    language_model._rope_deltas = None


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    # Stay scalar for one row; promote only when another row actually joins.
    cache_a = _ensure_batch_cache(cache_a)
    cache_b = _ensure_batch_cache(cache_b)
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _normalize_rope_deltas(rope_deltas: Any) -> Any:
    if rope_deltas is None:
        return None
    # Mirrors external/src/mlx-vlm/mlx_vlm/generate.py:
    # PromptProcessingBatch.generate.
    # Qwen-style language models expose scalar or (B,) deltas; batching wants
    # one row per sequence.
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
    # external/src/mlx-vlm/mlx_vlm/generate.py:GenerationBatch.extend
    # concatenates RoPE rows. We also synthesize zero rows so mixed image/text
    # batches do not broadcast an image row's MRoPE delta onto a text-only row.
    if rope_deltas is None:
        rope_deltas = _empty_rope_deltas_like(other_rope_deltas, count)
    if other_rope_deltas is None:
        other_rope_deltas = _empty_rope_deltas_like(rope_deltas, other_count)
    return mx.concatenate([rope_deltas, other_rope_deltas])


def _capture_rope_deltas(model, rows: int) -> Any | None:
    language_model = getattr(model, "language_model", model)
    if not hasattr(language_model, "_rope_deltas"):
        return None

    rope_deltas = getattr(language_model, "_rope_deltas", None)
    if rope_deltas is None:
        if str(getattr(language_model, "model_type", "")).startswith("qwen3_5"):
            return None
        return mx.zeros((rows, 1), dtype=mx.int32)

    rope_deltas = _normalize_rope_deltas(rope_deltas)
    if rope_deltas.shape[0] == 1 and rows > 1:
        rope_deltas = mx.broadcast_to(rope_deltas, (rows, 1))
    if rope_deltas.shape[0] != rows:
        raise RuntimeError(
            f"_rope_deltas shape {rope_deltas.shape} does not match batch size {rows}"
        )
    return rope_deltas


def _sample_with_samplers(
    logprobs: mx.array,
    samplers: list[Callable[[mx.array], mx.array]],
) -> mx.array:
    return mx.concatenate(
        [sampler(logprobs[i : i + 1]) for i, sampler in enumerate(samplers)],
        axis=0,
    )


def _apply_logits_processors(
    logits: mx.array,
    tokens: list[list[int]],
    logits_processors: list[list[LogitsProcessor]],
    last_tokens: mx.array | None = None,
) -> mx.array:
    processed_logits = []
    for i in range(logits.shape[0]):
        sample_logits = logits[i : i + 1]
        appended_last_token = False
        for processor in logits_processors[i]:
            if last_tokens is not None and isinstance(
                processor, RepetitionPenaltyProcessor
            ):
                sample_logits = processor.process_last_token(
                    last_tokens[i : i + 1], sample_logits
                )
            else:
                if last_tokens is not None and not appended_last_token:
                    tokens[i].append(last_tokens[i : i + 1].tolist()[0])
                    appended_last_token = True
                sample_logits = processor(mx.array(tokens[i]), sample_logits)
        processed_logits.append(sample_logits)
    return mx.concatenate(processed_logits, axis=0)


def _top_logprobs(
    logprobs: mx.array, k: int
) -> tuple[mx.array | None, mx.array | None]:
    if k <= 0:
        return None, None

    sort_idx = mx.argsort(logprobs, axis=-1)
    top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
    return top_idx, mx.take_along_axis(logprobs, top_idx, axis=-1)


def _pad_top_logprobs(
    top_idx: mx.array | None,
    top_logprobs: mx.array | None,
    rows: int,
    target_k: int,
    *,
    idx_dtype=mx.int32,
    logprob_dtype=mx.float32,
) -> tuple[mx.array | None, mx.array | None]:
    if target_k <= 0:
        return None, None
    if top_idx is None:
        return (
            mx.zeros((rows, target_k), dtype=idx_dtype),
            mx.zeros((rows, target_k), dtype=logprob_dtype),
        )
    if top_idx.shape[1] > target_k:
        return top_idx[:, :target_k], top_logprobs[:, :target_k]
    if top_idx.shape[1] == target_k:
        return top_idx, top_logprobs

    pad_k = target_k - top_idx.shape[1]
    return (
        mx.concatenate(
            [top_idx, mx.zeros((rows, pad_k), dtype=top_idx.dtype)],
            axis=1,
        ),
        mx.concatenate(
            [top_logprobs, mx.zeros((rows, pad_k), dtype=top_logprobs.dtype)],
            axis=1,
        ),
    )


def _sample_next_token(
    logprobs: mx.array,
    samplers: list[Callable[[mx.array], mx.array]],
    top_logprobs_k: int,
) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
    sampled = _sample_with_samplers(logprobs, samplers)
    token_logprobs = logprobs[mx.arange(sampled.shape[0]), sampled]
    top_idx, top_logprobs = _top_logprobs(logprobs, top_logprobs_k)
    return sampled, token_logprobs, top_idx, top_logprobs


def _materialize_step_outputs(
    tokens: mx.array,
    token_logprobs: mx.array | None,
    top_idx: mx.array | None,
    top_logprobs: mx.array | None,
) -> tuple[
    list[int], list[float] | None, list[list[int]] | None, list[list[float]] | None
]:
    if token_logprobs is None:
        mx.eval(tokens)
        return tokens.tolist(), None, None, None

    eval_targets = [tokens, token_logprobs]
    if top_idx is not None:
        eval_targets.extend([top_idx, top_logprobs])
    mx.eval(*eval_targets)

    return (
        tokens.tolist(),
        token_logprobs.tolist(),
        top_idx.tolist() if top_idx is not None else None,
        top_logprobs.tolist() if top_logprobs is not None else None,
    )


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
        inputs: mx.array | None,
        prompt_cache: list[Any],
        samplers: list[Callable[[mx.array], mx.array]],
        stop_criteria,
        max_tokens: list[int],
        logits_processors: list[list[LogitsProcessor]],
        prefix_cache_save_states: list[_PrefixCacheSaveState],
        top_logprobs_k: int = 0,
        top_logprobs: list[int] | None = None,
        all_tokens: Optional[list[list[int]]] = None,
        rope_deltas: Any | None = None,
    ):
        self.model = model
        self.prompt_cache = prompt_cache
        self.stop_criteria = stop_criteria
        top_logprobs_by_row = (
            list(top_logprobs)
            if top_logprobs is not None
            else [top_logprobs_k] * len(uids)
        )
        tokens_by_row = (
            [list(tokens) for tokens in all_tokens]
            if all_tokens is not None
            else [[] for _ in uids]
        )
        self._rows = [
            _GenerationRow(
                uid=uid,
                sampler=sampler,
                logits_processors=processors,
                max_tokens=max_token_count,
                top_logprobs=row_top_logprobs,
                tokens=tokens,
                prefix_cache_save_state=save_state,
            )
            for (
                uid,
                sampler,
                processors,
                max_token_count,
                row_top_logprobs,
                tokens,
                save_state,
            ) in zip(
                uids,
                samplers,
                logits_processors,
                max_tokens,
                top_logprobs_by_row,
                tokens_by_row,
                prefix_cache_save_states,
            )
        ]
        self.top_logprobs_k = max(
            (row.top_logprobs for row in self._rows),
            default=0,
        )

        self._current_tokens = None
        self._current_token_logprobs = None
        self._next_tokens = inputs
        self._next_token_logprobs = None
        self._next_top_idx = None
        self._next_top_logprobs = None
        self._rope_deltas = None
        self._rope_deltas = _normalize_rope_deltas(rope_deltas)

    def __len__(self):
        return len(self._rows)

    @property
    def uids(self) -> list[int]:
        return [row.uid for row in self._rows]

    def _step(self):
        # Decode-ahead: emit the token sampled last step while scheduling the next.
        self._current_tokens = self._next_tokens
        self._current_token_logprobs = self._next_token_logprobs
        prev_top_idx = self._next_top_idx
        prev_top_logprobs = self._next_top_logprobs
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            # Same handoff as external/src/mlx-vlm/mlx_vlm/generate.py:
            # GenerationBatch._step. Qwen language models consume this kwarg
            # when deriving decode position_ids from cache offsets; see
            # external/src/mlx-vlm/mlx_vlm/models/qwen3_5/language.py.
            fwd_kwargs["rope_deltas"] = self._rope_deltas
            _sync_scalar_rope_deltas(self.model, self.prompt_cache, self._rope_deltas)

        output = self.model(inputs[:, None], cache=self.prompt_cache, **fwd_kwargs)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        row_tokens = [row.tokens for row in self._rows]
        row_token_lens = [len(tokens) for tokens in row_tokens]
        row_logits_processors = [row.logits_processors for row in self._rows]
        if any(row_logits_processors):
            logits = _apply_logits_processors(
                logits,
                row_tokens,
                row_logits_processors,
                last_tokens=inputs,
            )

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled, sampled_logprobs, top_idx, top_logprobs = _sample_next_token(
            logprobs,
            [row.sampler for row in self._rows],
            self.top_logprobs_k,
        )

        self._next_tokens = sampled
        self._next_token_logprobs = sampled_logprobs
        self._next_top_idx = top_idx
        self._next_top_logprobs = top_logprobs

        eval_targets = [sampled, sampled_logprobs]
        if top_idx is not None:
            eval_targets.extend([top_idx, top_logprobs])
        mx.async_eval(*eval_targets)

        tokens, token_logprob_list, top_idx_list, top_logprob_list = (
            _materialize_step_outputs(
                inputs,
                self._current_token_logprobs,
                prev_top_idx,
                prev_top_logprobs,
            )
        )

        for seq_tokens, token, old_len in zip(row_tokens, tokens, row_token_lens):
            if len(seq_tokens) == old_len:
                seq_tokens.append(token)
        return tokens, token_logprob_list, top_idx_list, top_logprob_list

    def extract_cache(self, idx: int) -> list[Any] | None:
        if idx == 0 and _is_scalar_prompt_cache(self.prompt_cache):
            return list(self.prompt_cache)
        extracted = []
        for cache in self.prompt_cache:
            if not hasattr(cache, "extract") or getattr(cache, "keys", True) is None:
                return None
            extracted.append(cache.extract(idx))
        return extracted

    def extract_rope_deltas(self, idx: int) -> Any | None:
        if self._rope_deltas is None:
            return None
        # Keep one row with the prompt cache for exact hot-cache reuse. Disk
        # restores intentionally recompute RoPE side state during prompt prep.
        return mx.contiguous(self._rope_deltas[idx : idx + 1])

    def append_prefilled_sequence(self, prefilled: "GenerationBatch"):
        """Append the one sequence that just finished prompt prefill."""
        count = len(self)
        prefilled_count = len(prefilled)

        self._rows.extend(prefilled._rows)
        self.prompt_cache = _extend_cache(self.prompt_cache, prefilled.prompt_cache)
        next_top_logprobs_k = max(
            (row.top_logprobs for row in self._rows),
            default=0,
        )

        if self._current_tokens is None:
            self._current_tokens = prefilled._current_tokens
            self._current_token_logprobs = prefilled._current_token_logprobs
        elif prefilled._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, prefilled._current_tokens]
            )
            if (
                self._current_token_logprobs is not None
                and prefilled._current_token_logprobs is not None
            ):
                self._current_token_logprobs = mx.concatenate(
                    [
                        self._current_token_logprobs,
                        prefilled._current_token_logprobs,
                    ]
                )

        if self._next_tokens is None:
            self._next_tokens = prefilled._next_tokens
            self._next_token_logprobs = prefilled._next_token_logprobs
            self._next_top_idx = prefilled._next_top_idx
            self._next_top_logprobs = prefilled._next_top_logprobs
        elif prefilled._next_tokens is not None:
            self._next_tokens = mx.concatenate(
                [self._next_tokens, prefilled._next_tokens]
            )
            if (
                self._next_token_logprobs is not None
                and prefilled._next_token_logprobs is not None
            ):
                self._next_token_logprobs = mx.concatenate(
                    [self._next_token_logprobs, prefilled._next_token_logprobs]
                )

            self._next_top_idx, self._next_top_logprobs = _pad_top_logprobs(
                self._next_top_idx,
                self._next_top_logprobs,
                count,
                next_top_logprobs_k,
            )
            prefilled_top_idx, prefilled_top_logprobs = _pad_top_logprobs(
                prefilled._next_top_idx,
                prefilled._next_top_logprobs,
                prefilled_count,
                next_top_logprobs_k,
            )
            if self._next_top_idx is not None:
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, prefilled_top_idx]
                )
                self._next_top_logprobs = mx.concatenate(
                    [self._next_top_logprobs, prefilled_top_logprobs]
                )

        self._rope_deltas = _extend_optional_rope_deltas(
            self._rope_deltas,
            count,
            prefilled._rope_deltas,
            prefilled_count,
        )

        self.top_logprobs_k = next_top_logprobs_k

    def filter(self, keep: list[int]):
        self._rows = [self._rows[idx] for idx in keep]
        self.top_logprobs_k = max(
            (row.top_logprobs for row in self._rows),
            default=0,
        )

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_token_logprobs = None
            self._next_tokens = None
            self._next_token_logprobs = None
            self._next_top_idx = None
            self._next_top_logprobs = None
            self._rope_deltas = None
            return

        keep_arr = mx.array(keep, mx.int32)
        for cache in self.prompt_cache:
            if hasattr(cache, "filter"):
                cache.filter(keep_arr)
        if len(keep) == 1 and not _is_scalar_prompt_cache(self.prompt_cache):
            extracted = self.extract_cache(0)
            if extracted is not None:
                self.prompt_cache = extracted
        if self._next_tokens is not None:
            self._next_tokens = self._next_tokens[keep_arr]
        if self._next_token_logprobs is not None:
            self._next_token_logprobs = self._next_token_logprobs[keep_arr]
        if self._next_top_idx is not None:
            self._next_top_idx = self._next_top_idx[keep_arr]
            self._next_top_logprobs = self._next_top_logprobs[keep_arr]
            self._next_top_idx, self._next_top_logprobs = _pad_top_logprobs(
                self._next_top_idx,
                self._next_top_logprobs,
                len(keep),
                self.top_logprobs_k,
            )
        if self._rope_deltas is not None:
            # Matches external/src/mlx-vlm/mlx_vlm/generate.py:
            # GenerationBatch.filter:
            # RoPE rows are batch-aligned with cache rows.
            self._rope_deltas = self._rope_deltas[keep_arr]

    def _emit_cache_save_snapshot(self, idx: int) -> None:
        try:
            row = self._rows[idx]
            save_state = row.prefix_cache_save_state
            if save_state.callback is None:
                return

            current_len = len(row.tokens)
            chunks = save_state.chunks
            next_chunk_idx = save_state.next_chunk_idx
            next_chunk_end = (
                chunks[next_chunk_idx].end
                if next_chunk_idx < len(chunks)
                else (chunks[-1].end if chunks else 0) + DEFAULT_PREFIX_CHUNK_SIZE
            )
            if current_len < next_chunk_end:
                return

            extend_prefix_cache_chunks(
                row.tokens,
                save_state.image_spans,
                chunks,
            )
            end_chunk_idx = len(chunks)
            save_state.next_chunk_idx = end_chunk_idx

            prompt_cache = self.extract_cache(idx)
            if prompt_cache is None:
                return
            save_state.callback(
                prompt_cache,
                chunks,
                next_chunk_idx,
                end_chunk_idx,
                current_len,
            )
        except Exception:
            logger.debug("Skipping decode prompt-cache snapshot.", exc_info=True)

    def next(self) -> list[Response]:
        if not self._rows:
            return []

        tokens, token_logprob_list, top_idx_list, top_logprob_list = self._step()

        keep = []
        responses = []
        for i, row in enumerate(self._rows):
            finish_reason = None
            row.num_tokens += 1
            tok = tokens[i]

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif row.num_tokens >= row.max_tokens:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            self._emit_cache_save_snapshot(i)

            top_logprobs = None
            if row.top_logprobs > 0 and top_idx_list is not None:
                top_logprobs = list(
                    zip(
                        top_idx_list[i][: row.top_logprobs],
                        top_logprob_list[i][: row.top_logprobs],
                    )
                )

            responses.append(
                self.Response(
                    uid=row.uid,
                    token=tok,
                    token_logprob=(
                        token_logprob_list[i] if token_logprob_list is not None else 0.0
                    ),
                    finish_reason=finish_reason,
                    top_logprobs=top_logprobs,
                    prompt_cache=self.extract_cache(i) if finish_reason else None,
                    all_tokens=list(row.tokens) if finish_reason else None,
                    rope_deltas=self.extract_rope_deltas(i) if finish_reason else None,
                )
            )

        if len(keep) < len(self._rows):
            self.filter(keep)

        return responses

    @classmethod
    def empty(cls, model, stop_criteria, top_logprobs_k=0):
        return cls(
            model=model,
            uids=[],
            inputs=None,
            prompt_cache=[],
            samplers=[],
            stop_criteria=stop_criteria,
            max_tokens=[],
            top_logprobs_k=top_logprobs_k,
            top_logprobs=[],
            logits_processors=[],
            prefix_cache_save_states=[],
        )


class _PromptPrefill:
    """One request being prefetched before it joins the decode batch."""

    @dataclass
    class Response:
        uid: int
        progress: tuple[int, int]

    def __init__(
        self,
        model: nn.Module,
        uid: int,
        input_ids: list[int],
        max_tokens: int,
        top_logprobs: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: list[LogitsProcessor],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        prefix_cache_save_state: _PrefixCacheSaveState,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        cache: Optional[list[Any]] = None,
        all_tokens: Optional[list[int]] = None,
        rope_deltas: Any | None = None,
    ):
        self.model = model
        self.uid = uid
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.sampler = sampler
        self.logits_processors = logits_processors
        self.prefill_step_size = prefill_step_size

        self._input_ids = _left_pad_prompts([input_ids], max_length=len(input_ids))
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs
        self._prompt_token_ids = list(input_ids)
        self._all_tokens = list(all_tokens) if all_tokens is not None else []
        self._processed_prefix_len = len(self._all_tokens)
        self._prefix_cache_save_state = prefix_cache_save_state
        self._visual_prefill_prefix_len = gemma4_visual_prefill_prefix_len(
            model,
            prompt_kwargs,
            prefix_cache_save_state.image_spans,
            len(self._all_tokens),
        )
        # Restored exact hot caches may carry an existing row. Trimmed/disk
        # restores pass None and let prompt prep recompute it.
        self._rope_deltas = _normalize_rope_deltas(rope_deltas)

        if cache is None:
            # Prompt prefill handles one request, so scalar caches avoid batch overhead.
            self.prompt_cache = make_prompt_cache(model)
        else:
            self.prompt_cache = cache

    def progress_responses(self) -> list[Response]:
        total = len(self._all_tokens) + len(self._prompt_token_ids)
        return [
            self.Response(
                self.uid,
                (self._processed_prefix_len, total),
            )
        ]

    def needs_processing(self):
        return self._next_prompt_step_size() > 0

    def prompt_step(self) -> int:
        n = self._next_prompt_step_size()
        prompt_kwargs = self._prompt_kwargs_for_next(n)
        # Prompt kwargs with explicit MRoPE state belong to an image prompt; otherwise
        # this text-only chunk must not inherit state from the active decode batch.
        _clear_qwen3_5_text_rope_state(self.model, prompt_kwargs)
        try:
            self.model(
                self._input_ids[:, :n],
                cache=self.prompt_cache,
                inputs_embeds=self._inputs_embeds[:, :n],
                **prompt_kwargs,
            )
        finally:
            _clear_qwen3_5_text_rope_state(self.model, prompt_kwargs)
        mx.eval([c.state for c in self.prompt_cache])
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        self._drop_processed_prompt_kwargs(n)
        self._processed_prefix_len += n
        self._emit_cache_save_snapshots()
        mx.clear_cache()
        return n

    def _next_prompt_step_size(self) -> int:
        """Return the next chunked prefill size.

        When disk checkpointing is active, a restored prefix can start between
        normal prefill boundaries, so the first step may be short to land back
        on the regular step grid.
        """
        if self.prefill_step_size is None:
            return 0

        remaining_tokens = self._inputs_embeds.shape[1]
        if remaining_tokens <= 1:
            return 0

        # Apply the Gemma4 visual-prefix rule before normal cache-boundary
        # alignment. If a restore lands before a new image, token-type padding
        # lets the image suffix build its mask against the restored KV prefix.
        visual_prefix_remaining = self._visual_prefill_prefix_remaining()
        if visual_prefix_remaining is not None:
            if visual_prefix_remaining < remaining_tokens:
                return visual_prefix_remaining
            return 0

        saving_prompt_cache = self._prefix_cache_save_state.callback is not None
        processed_remainder = self._processed_prefix_len % self.prefill_step_size
        if saving_prompt_cache and processed_remainder:
            # After a partial restore, land on the next normal prefill boundary.
            alignment_step = self.prefill_step_size - processed_remainder
            if alignment_step < remaining_tokens:
                return min(alignment_step, remaining_tokens - 1)

        if remaining_tokens > self.prefill_step_size:
            return min(self.prefill_step_size, remaining_tokens - 1)

        if saving_prompt_cache and any(
            type(cache).__name__ == "ArraysCache" for cache in self.prompt_cache
        ):
            # Opaque state caches are restorable only at exact saved boundaries.
            max_reusable_prefix_len = self._processed_prefix_len + remaining_tokens - 1
            target_prefix_len = (
                max_reusable_prefix_len // DEFAULT_PREFIX_CHUNK_SIZE
            ) * DEFAULT_PREFIX_CHUNK_SIZE
            if target_prefix_len > self._processed_prefix_len:
                return target_prefix_len - self._processed_prefix_len

        return 0

    def _visual_prefill_prefix_remaining(self) -> int | None:
        if self._visual_prefill_prefix_len is None:
            return None
        processed_prompt_len = self._processed_prefix_len - len(self._all_tokens)
        remaining = self._visual_prefill_prefix_len - processed_prompt_len
        if remaining <= 0:
            return None
        return remaining

    def _prompt_kwargs_for_next(self, n: int) -> dict:
        # Slice locally instead of relying on model-specific n_to_process hacks.
        prompt_kwargs = slice_prompt_kwargs(
            self._prompt_kwargs,
            0,
            n,
            mask_key_end=self._processed_prefix_len + n,
        )
        return self._prepare_cached_suffix_prompt_kwargs(
            prompt_kwargs,
            self._processed_prefix_len + n,
        )

    def _drop_processed_prompt_kwargs(self, n: int) -> None:
        self._prompt_kwargs = drop_prompt_kwargs_prefix(self._prompt_kwargs, n)

    def _emit_cache_save_snapshots(self) -> None:
        try:
            save_state = self._prefix_cache_save_state
            if save_state.callback is None:
                return

            processed = self._processed_prefix_len
            start_chunk_idx = save_state.next_chunk_idx
            end_chunk_idx = min(
                processed // DEFAULT_PREFIX_CHUNK_SIZE,
                len(save_state.chunks),
            )
            if start_chunk_idx >= end_chunk_idx:
                return

            save_state.next_chunk_idx = end_chunk_idx
            # One large prefill can backfill several crossed cache chunks.
            save_state.callback(
                [
                    cache.extract(0) if hasattr(cache, "extract") else cache
                    for cache in self.prompt_cache
                ],
                save_state.chunks,
                start_chunk_idx,
                end_chunk_idx,
                processed,
            )
        except Exception:
            logger.debug("Skipping prefill prompt-cache snapshot.", exc_info=True)

    def generate(self, stop_criteria) -> tuple[GenerationBatch, list[Response]]:
        # This final prompt pass runs after active batched decode in the same tick.
        prompt_kwargs = self._prompt_kwargs_for_final()
        _clear_qwen3_5_text_rope_state(self.model, prompt_kwargs)
        try:
            output = self.model(
                self._input_ids,
                cache=self.prompt_cache,
                inputs_embeds=self._inputs_embeds,
                **prompt_kwargs,
            )
        finally:
            _clear_qwen3_5_text_rope_state(self.model, prompt_kwargs)
        self._processed_prefix_len = len(self._all_tokens) + len(self._prompt_token_ids)
        self._emit_cache_save_snapshots()
        prompt_responses = self.progress_responses()
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]
        if self.logits_processors:
            logits = _apply_logits_processors(
                logits,
                [self._all_tokens + self._prompt_token_ids],
                [self.logits_processors],
            )
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens, first_logprobs, top_idx, top_logprobs = _sample_next_token(
            logprobs, [self.sampler], self.top_logprobs
        )

        gen_batch = GenerationBatch(
            model=self.model,
            uids=[self.uid],
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            samplers=[self.sampler],
            stop_criteria=stop_criteria,
            max_tokens=[self.max_tokens],
            top_logprobs=[self.top_logprobs],
            all_tokens=[self._all_tokens + self._prompt_token_ids],
            rope_deltas=self._rope_deltas,
            logits_processors=[self.logits_processors],
            prefix_cache_save_states=[self._prefix_cache_save_state],
        )
        gen_batch._next_token_logprobs = first_logprobs
        gen_batch._next_top_idx = top_idx
        gen_batch._next_top_logprobs = top_logprobs

        # Same capture point as external/src/mlx-vlm/mlx_vlm/generate.py:
        # PromptProcessingBatch.generate. Qwen model get_input_embeddings
        # primes language_model._rope_deltas via get_rope_index; see
        # external/src/mlx-vlm/mlx_vlm/models/qwen3_5/qwen3_5.py.
        rope_deltas = _capture_rope_deltas(self.model, len(gen_batch.uids))
        gen_batch._rope_deltas = rope_deltas

        self.prompt_cache = []
        return gen_batch, prompt_responses

    def _prompt_kwargs_for_final(self) -> dict:
        prompt_kwargs = slice_prompt_kwargs(
            self._prompt_kwargs,
            0,
            self._inputs_embeds.shape[1],
            mask_key_end=self._processed_prefix_len + self._inputs_embeds.shape[1],
        )
        return self._prepare_cached_suffix_prompt_kwargs(
            prompt_kwargs,
            self._processed_prefix_len + self._inputs_embeds.shape[1],
        )

    def _prepare_cached_suffix_prompt_kwargs(
        self,
        prompt_kwargs: dict,
        key_len: int,
    ) -> dict:
        if self._visual_prefill_prefix_len is None:
            return prompt_kwargs
        return prepare_gemma4_cached_suffix_prompt_kwargs(prompt_kwargs, key_len)


class BatchGenerator:
    """MLX-VLM-style continuous batcher with single-request prompt prefill."""

    def __init__(
        self,
        model,
        stop_criteria,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        top_logprobs_k: int = 0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._default_top_logprobs = top_logprobs_k
        self.stop_criteria = stop_criteria
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.completion_batch_size = completion_batch_size

        self._prompt_batch: Optional[_PromptPrefill] = None
        self._unprocessed_sequences: list[_PendingSequence] = []

        self._steps_counter = 0

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model))
        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.stop_criteria,
        )

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompt: list[int],
        *,
        inputs_embeds: mx.array,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: list[LogitsProcessor],
        prompt_kwargs: dict,
        prefix_cache_chunks: list[PromptPrefixChunk],
        image_spans: list[PromptImageSpan],
        max_tokens: int | None = None,
        top_logprobs: int | None = None,
        cache: Optional[list[Any]] = None,
        all_tokens: list[int],
        rope_deltas: Any | None = None,
        next_prefix_cache_chunk_idx: int,
        prompt_cache_save_callback: Optional[PromptCacheSaveCallback] = None,
    ) -> int:
        uid = self.uid_count
        self.uid_count += 1
        self._unprocessed_sequences.append(
            _PendingSequence(
                uid=uid,
                prompt=prompt,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                top_logprobs=(
                    self._default_top_logprobs if top_logprobs is None else top_logprobs
                ),
                sampler=sampler,
                # Stateful processors are request-owned; callers must not share them.
                logits_processors=list(logits_processors),
                inputs_embeds=inputs_embeds,
                cache=cache,
                all_tokens=list(all_tokens),
                rope_deltas=rope_deltas,
                prompt_kwargs=prompt_kwargs,
                prefix_cache_save_state=_PrefixCacheSaveState(
                    chunks=list(prefix_cache_chunks),
                    next_chunk_idx=next_prefix_cache_chunk_idx,
                    image_spans=list(image_spans),
                    callback=prompt_cache_save_callback,
                ),
            )
        )
        return uid

    def next(self):
        return self._next()

    def remove(self, uid) -> bool:
        for i, sequence in enumerate(self._unprocessed_sequences):
            if sequence.uid == uid:
                self._unprocessed_sequences.pop(i)
                return True

        if self._prompt_batch is not None and uid == self._prompt_batch.uid:
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
                prompt_responses = self._prompt_batch.progress_responses()
                return prompt_responses, generation_responses

            gen_batch, prompt_responses = self._prompt_batch.generate(
                self.stop_criteria
            )
            self._generation_batch.append_prefilled_sequence(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= 1:
            sequence = self._unprocessed_sequences.pop(0)

            self._prompt_batch = _PromptPrefill(
                model=self.model,
                uid=sequence.uid,
                input_ids=sequence.prompt,
                max_tokens=sequence.max_tokens,
                top_logprobs=sequence.top_logprobs,
                sampler=sequence.sampler,
                logits_processors=sequence.logits_processors,
                inputs_embeds=sequence.inputs_embeds,
                prompt_kwargs=sequence.prompt_kwargs,
                prefix_cache_save_state=sequence.prefix_cache_save_state,
                prefill_step_size=self.prefill_step_size,
                cache=sequence.cache,
                all_tokens=sequence.all_tokens,
                rope_deltas=sequence.rope_deltas,
            )

            if self._prompt_batch.needs_processing():
                self._prompt_batch.prompt_step()
                prompt_responses = self._prompt_batch.progress_responses()
            else:
                gen_batch, prompt_responses = self._prompt_batch.generate(
                    self.stop_criteria
                )
                self._generation_batch.append_prefilled_sequence(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses
