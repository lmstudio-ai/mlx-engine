"""Fit a model's context length to the available unified memory.

A model's native context limit does not account for its loaded weights or the
memory needed to read a long prompt. Guessing by allocating progressively larger
prompts is unsafe because a Metal out-of-memory error can terminate the process.
Instead, a one-token probe reveals the loaded model's real cache shapes and
dtypes, then uses the formula below. The formula is validated for Gemma 4 and
Qwen 3.5, and is used as a best-effort fit for other measurable cache layouts.

    fixed = loaded baseline + rotating cache + recurrent state
    bytes/token = full KV + prompt embedding + attention workspace
    context = (recommended working set - reserve - fixed) / bytes/token

Long-prefill experiments on dense and MoE Qwen and Gemma models showed that
attention memory matched `query heads * prefill chunk * activation dtype size`.
The largest measured peak not explained by the formula was 2.41 GiB,
motivating the 3 GiB reserve floor.

For example, Qwen 3.6 35B-A3B on a 27 GiB working set has about 19.06 GiB of
fixed memory and uses 88 KiB per token. After the 3 GiB reserve, the formula
fits 58,846 tokens, which rounds to a 58,880-token cache boundary.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

logger = logging.getLogger(__name__)


GIB = 1024**3
# Product minimum; the fitted value is still capped by the model's native limit.
MIN_FITTED_CONTEXT_TOKENS = 4_096
# Round the largest 2.41 GiB unexplained experimental peak up to 3 GiB.
MIN_RUNTIME_RESERVE_BYTES = 3 * GIB

_FAMILY_GEMMA4 = "gemma4"
_FAMILY_QWEN3_5 = "qwen3_5"
_SUPPORTED_CACHE_TYPES = {"KVCache", "RotatingKVCache", "ArraysCache"}


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


@dataclass(frozen=True)
class CacheFitProfile:
    family: str
    allocation_step: int
    full_kv_bytes_per_token: int
    prompt_embedding_bytes_per_token: int
    query_attention_heads: int
    activation_dtype_bytes: int
    prefill_step_size: int
    rotating_peak_bytes: int
    fixed_ssm_bytes: int
    max_context_length: int


@dataclass(frozen=True)
class ContextFitResult:
    context_length: int
    runtime_reserve_bytes: int
    safe_ceiling_bytes: int


def fit_batched_vlm_context(
    *,
    model: Any,
    prefill_step_size: int,
) -> int | None:
    """Return a fitted token limit, or `None` to leave the limit unchanged.

    A one-token probe measures the model's cache, embedding, and activation
    layout. Gemma 4 and Qwen 3.5 use the validated fit. Other families use the
    same fit when their memory layout can be measured without assumptions.
    """
    max_context_length = None
    validated_family = False
    try:
        language_model = getattr(model, "language_model", model)
        language_config = getattr(language_model, "config", None)
        language_args = getattr(language_model, "args", None)
        model_config = getattr(model, "config", None)
        text_config = _config_value(model_config, "text_config")
        config_sources = (
            language_config,
            language_args,
            text_config,
            model_config,
        )
        model_type = getattr(language_model, "model_type", None)
        for source in config_sources:
            if model_type is not None:
                break
            model_type = _config_value(source, "model_type")
        model_type = str(model_type or "unknown").lower()
        if model_type.startswith("gemma4"):
            family = _FAMILY_GEMMA4
            validated_family = True
        elif model_type.startswith(("qwen3_5", "qwen3_6")):
            family = _FAMILY_QWEN3_5
            validated_family = True
        else:
            family = model_type

        for source in config_sources:
            max_context_length = _config_value(source, "max_position_embeddings")
            if max_context_length is not None:
                break
        if max_context_length is None:
            logger.error(
                "Model context auto-fit could not find the native context; "
                "leaving context unchanged"
            )
            return None
        if not validated_family and max_context_length <= MIN_FITTED_CONTEXT_TOKENS:
            logger.info(
                "Model family %s reports a %s token maximum; leaving context unchanged",
                family,
                f"{max_context_length:,}",
            )
            return None

        try:
            profile = _probe_cache_fit_profile(
                model=model,
                language_model=language_model,
                family=family,
                max_context_length=max_context_length,
                prefill_step_size=prefill_step_size,
            )
        finally:
            # Release the temporary probe arrays before measuring the loaded model.
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
            mx.synchronize()

        if profile is None:
            return None

        # Active arrays and allocator cache both occupy unified memory. Together
        # they are the starting cost before the prompt grows.
        baseline_bytes = mx.get_active_memory() + mx.get_cache_memory()
        # Apple's recommended working set is the process budget, not total RAM.
        working_set_bytes = mx.device_info()["max_recommended_working_set_size"]
        result = calculate_context_fit(
            profile,
            working_set_bytes=working_set_bytes,
            baseline_bytes=baseline_bytes,
        )
        if not validated_family and result.context_length <= MIN_FITTED_CONTEXT_TOKENS:
            logger.info(
                "Best-effort context fit for model family %s was only %s tokens; "
                "leaving context unchanged",
                family,
                f"{result.context_length:,}",
            )
            return None

        attention_workspace_bytes_per_token = (
            profile.query_attention_heads
            * profile.prefill_step_size
            * profile.activation_dtype_bytes
        )
        peak_bytes_per_token = (
            profile.full_kv_bytes_per_token
            + profile.prompt_embedding_bytes_per_token
            + attention_workspace_bytes_per_token
        )
        estimated_memory_bytes = (
            baseline_bytes
            + profile.rotating_peak_bytes
            + profile.fixed_ssm_bytes
            + peak_bytes_per_token * result.context_length
        )
        logger.info(
            "Model context auto-fit: family=%s max=%s fitted=%s "
            "working_set=%.2fGiB reserve=%.2fGiB safe_ceiling=%.2fGiB "
            "baseline=%.2fGiB full_kv=%dB/token embedding=%dB/token "
            "attention=%dB/token rotating_peak=%.2fGiB fixed_ssm=%.2fGiB "
            "estimated_peak=%.2fGiB",
            profile.family,
            f"{max_context_length:,}",
            f"{result.context_length:,}",
            working_set_bytes / GIB,
            result.runtime_reserve_bytes / GIB,
            result.safe_ceiling_bytes / GIB,
            baseline_bytes / GIB,
            profile.full_kv_bytes_per_token,
            profile.prompt_embedding_bytes_per_token,
            attention_workspace_bytes_per_token,
            profile.rotating_peak_bytes / GIB,
            profile.fixed_ssm_bytes / GIB,
            estimated_memory_bytes / GIB,
        )
        return result.context_length
    except Exception:
        logger.exception("Model context auto-fit failed; leaving context unchanged")
        return None


def calculate_context_fit(
    profile: CacheFitProfile,
    *,
    working_set_bytes: int,
    baseline_bytes: int,
) -> ContextFitResult:
    """Calculate the context limit from the model's measured memory costs.

    The fit subtracts the runtime reserve and fixed memory from the recommended
    working set. It divides what remains by the per-token peak for KV, retained
    prompt embeddings, and chunked attention work. The result contains the
    chosen token limit and the reserve and safe ceiling used to calculate it.
    """
    # The modeled terms undercounted measured prefill peaks by at most 2.41 GiB.
    # Leave 3 GiB for that variation, or 5% on very large working sets.
    runtime_reserve_bytes = max(MIN_RUNTIME_RESERVE_BYTES, working_set_bytes // 20)
    safe_ceiling_bytes = working_set_bytes - runtime_reserve_bytes
    # These costs do not grow with the ordinary full-context KV cache.
    fixed_memory_bytes = (
        baseline_bytes + profile.rotating_peak_bytes + profile.fixed_ssm_bytes
    )

    # Long-prefill measurements matched one activation value per query head and
    # prefill token for every context token. The dtype size is measured at runtime
    # rather than assuming the tested models' two-byte BF16/FP16 activations.
    attention_workspace_bytes_per_token = (
        profile.query_attention_heads
        * profile.prefill_step_size
        * profile.activation_dtype_bytes
    )
    peak_bytes_per_token = (
        profile.full_kv_bytes_per_token
        + profile.prompt_embedding_bytes_per_token
        + attention_workspace_bytes_per_token
    )

    # First find how many bytes remain for the prompt after paying the model's
    # fixed costs. If those costs already exceed the ceiling, zero bytes remain.
    available_prompt_bytes = max(0, safe_ceiling_bytes - fixed_memory_bytes)

    # Dividing the remaining bytes by one token's peak cost gives the number of
    # prompt tokens that fit in memory.
    tokens_that_fit = available_prompt_bytes // peak_bytes_per_token

    # MLX allocates KV in token blocks. Round to the measured block boundary;
    # the reserve absorbs the less-than-one-block difference.
    allocation_step = profile.allocation_step
    context_length = (
        (tokens_that_fit + allocation_step - 1) // allocation_step * allocation_step
    )
    if context_length < MIN_FITTED_CONTEXT_TOKENS:
        logger.warning(
            "Model context auto-fit calculated %s tokens; using the %s token minimum",
            f"{context_length:,}",
            f"{MIN_FITTED_CONTEXT_TOKENS:,}",
        )
        context_length = MIN_FITTED_CONTEXT_TOKENS
    # Available memory never permits extending the model beyond its native limit.
    context_length = min(profile.max_context_length, context_length)

    return ContextFitResult(
        context_length=context_length,
        runtime_reserve_bytes=runtime_reserve_bytes,
        safe_ceiling_bytes=safe_ceiling_bytes,
    )


def _probe_cache_fit_profile(
    *,
    model: Any,
    language_model: Any,
    family: str,
    max_context_length: int,
    prefill_step_size: int,
) -> CacheFitProfile | None:
    prompt_cache = make_prompt_cache(language_model)
    # One token is enough to allocate every cache in its real shape and dtype.
    input_ids = mx.zeros((1, 1), dtype=mx.int32)
    embedding_kwargs = {
        key: value
        for key, value in model.get_input_embeddings(input_ids).to_dict().items()
        if value is not None
    }
    inputs_embeds = embedding_kwargs.pop("inputs_embeds")
    # With one token, embedding nbytes is directly the retained bytes/token;
    # itemsize gives the activation dtype size used by attention.
    prompt_embedding_bytes_per_token = inputs_embeds.nbytes
    activation_dtype_bytes = inputs_embeds.itemsize

    language_config = getattr(language_model, "config", None)
    language_args = getattr(language_model, "args", None)
    query_attention_heads = _config_value(
        language_config, "num_attention_heads"
    ) or _config_value(language_args, "num_attention_heads")
    if query_attention_heads is None:
        query_attention_heads = _config_value(
            language_config, "n_heads"
        ) or _config_value(language_args, "n_heads")

    language_model(
        input_ids,
        cache=prompt_cache,
        inputs_embeds=inputs_embeds,
        **embedding_kwargs,
    )
    mx.eval([cache.state for cache in prompt_cache])
    mx.synchronize()

    # Full KV grows with context. Rotating KV and Qwen SSM state are fixed costs.
    full_kv_bytes_per_token = 0
    rotating_peak_bytes = 0
    fixed_ssm_bytes = 0
    # KV caches grow in token blocks. Every layer must use the same block size so
    # one rounded context length describes the whole model.
    cache_allocation_steps = set()

    for cache in prompt_cache:
        cache_type = type(cache).__name__
        if cache_type not in _SUPPORTED_CACHE_TYPES:
            logger.error(
                "Unsupported %s in %s cache topology; skipping context auto-fit",
                cache_type,
                family,
            )
            return None

        if cache_type == "KVCache":
            # nbytes includes both keys and values. Divide by the allocated token
            # width to get the cost of one token.
            full_kv_bytes_per_token += cache.nbytes // cache.keys.shape[2]
            cache_allocation_steps.add(cache.step)
        elif cache_type == "RotatingKVCache":
            if cache.keep != 0:
                logger.error(
                    "Rotating cache in %s uses keep=%s; skipping context auto-fit",
                    family,
                    cache.keep,
                )
                return None
            # One layer can briefly hold its window plus the new prefill chunk,
            # with one overlapping token: max_size + prefill_step_size - 1.
            rotating_peak_bytes += (
                cache.nbytes
                // cache.keys.shape[2]
                * (cache.max_size + prefill_step_size - 1)
            )
            cache_allocation_steps.add(cache.step)
        else:
            # ArraysCache holds fixed-size convolution or recurrent state.
            if any(state is None for state in cache.cache):
                logger.error(
                    "ArraysCache probe for %s left state uninitialized; "
                    "skipping context auto-fit",
                    family,
                )
                return None
            fixed_ssm_bytes += cache.nbytes

    if full_kv_bytes_per_token == 0:
        logger.error(
            "Model cache for %s has no full KV layers; skipping context auto-fit",
            family,
        )
        return None
    if len(cache_allocation_steps) != 1:
        logger.error(
            "Model cache for %s uses inconsistent allocation steps; "
            "skipping context auto-fit",
            family,
        )
        return None

    return CacheFitProfile(
        family=family,
        allocation_step=next(iter(cache_allocation_steps)),
        full_kv_bytes_per_token=full_kv_bytes_per_token,
        prompt_embedding_bytes_per_token=prompt_embedding_bytes_per_token,
        query_attention_heads=query_attention_heads,
        activation_dtype_bytes=activation_dtype_bytes,
        prefill_step_size=prefill_step_size,
        rotating_peak_bytes=rotating_peak_bytes,
        fixed_ssm_bytes=fixed_ssm_bytes,
        max_context_length=max_context_length,
    )
