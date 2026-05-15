"""
Sampling helpers shared by sequential and batched generation.
"""

import math
from typing import Optional

import mlx.core as mx


@mx.compile
def _apply_top_k(
    logprobs: mx.array,
    top_k: int,
) -> mx.array:
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(
        logprobs,
        mask_idx,
        mx.array(-float("inf"), logprobs.dtype),
        axis=-1,
    )


@mx.compile
def _apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            mx.array(False, tokens_to_remove.dtype),
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


@mx.compile
def _apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


def create_sampler(
    temp: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    min_tokens_to_keep: Optional[int],
    top_k: Optional[int],
):
    temp = 0.0 if temp is None else temp
    top_p = 0.0 if top_p is None else top_p
    min_p = 0.0 if min_p is None else min_p
    min_tokens_to_keep = 1 if min_tokens_to_keep is None else min_tokens_to_keep
    top_k = 0 if top_k is None else top_k

    if temp == 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    # Avoid mlx_lm.make_sampler's module-global compiled random helpers: they
    # capture thread-local PRNG state before batched worker threads call them.
    def sampler(logprobs):
        if top_p > 0 and top_p < 1.0:
            logprobs = _apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = _apply_min_p(logprobs, min_p, min_tokens_to_keep)
        if top_k > 0:
            logprobs = _apply_top_k(logprobs, top_k)
        return mx.random.categorical(logprobs * (1 / temp))

    return sampler
