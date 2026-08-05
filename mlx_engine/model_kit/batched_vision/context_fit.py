"""Fit context and precompute memory-aware prompt-prefill schedules.

A one-token probe measures the loaded model's real cache shapes and dtypes. For
a prepared prompt of ``P`` tokens, a post-chunk cache length ``K``, and prefill
step ``S``, the peak estimate is:

    baseline + fixed SSM + rotating_constant + rotating_per_step * S
    + prompt_input_bytes_per_token * P
    + full_kv_bytes_per_token * K
    + attention_bytes_per_context_per_step * S * K
    + runtime reserve

The fitted maximum is solved once at load. At request admission, the same
coefficients are solved once for cache-length boundaries and compiled into an
immutable schedule: use the configured step (normally 2,048) while it fits,
then 1,024, and use 512 only near the fitted maximum. Prompt callbacks only
look up those boundaries; they do not rerun the memory formula.

The fixed 3 GiB reserve covers the largest measured residual between this
formula and actual prefill peaks. Measurements showed that residual follows
model execution rather than installed RAM, so the reserve does not grow on
larger-memory Macs and is not reduced to force a larger minimum context.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_vlm.models.cache import make_prompt_cache

from mlx_engine.model_kit.batched_vision.prefill_plan import (
    PrefillPlan,
    PrefillSegment,
)

logger = logging.getLogger(__name__)


GIB = 1024**3
# Product minimum; the fitted value is still capped by the model's native limit.
MIN_FITTED_CONTEXT_TOKENS = 4_096
# Round the largest 2.41 GiB unexplained experimental peak up to 3 GiB.
MIN_RUNTIME_RESERVE_BYTES = 3 * GIB
PREFILL_STEP_CANDIDATES = (2_048, 1_024, 512)
# Bionic compacts at 15/16. Starting the smallest chunks no earlier than 85%
# of the reported maximum limits them to about 9.3% of the pre-compaction range.
SMALLEST_STEP_START_CONTEXT_PERCENT = 85

_FAMILY_GEMMA4 = "gemma4"
_FAMILY_GPT_OSS = "gpt_oss"
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
    prompt_input_bytes_per_token: int
    materializes_attention_scores: bool
    query_attention_heads: int
    activation_dtype_bytes: int
    prefill_step_size: int
    rotating_constant_bytes: int
    rotating_bytes_per_prefill_token: int
    fixed_ssm_bytes: int
    max_context_length: int

    @property
    def rotating_peak_bytes(self) -> int:
        return self.rotating_peak_bytes_for_step(self.prefill_step_size)

    def rotating_peak_bytes_for_step(self, prefill_step_size: int) -> int:
        return (
            self.rotating_constant_bytes
            + self.rotating_bytes_per_prefill_token * prefill_step_size
        )

    @property
    def attention_bytes_per_context_per_prefill_token(self) -> int:
        if not self.materializes_attention_scores:
            return 0
        return self.query_attention_heads * self.activation_dtype_bytes


@dataclass(frozen=True)
class ContextFitResult:
    context_length: int
    runtime_reserve_bytes: int
    safe_ceiling_bytes: int
    profile: CacheFitProfile | None = None
    baseline_bytes: int = 0
    working_set_bytes: int = 0
    prefill_step_sizes: tuple[int, ...] = ()
    step_context_limits: tuple[tuple[int, int], ...] = ()

    def make_request_prefill_plan(
        self,
        *,
        prompt_context_length: int,
    ) -> PrefillPlan | None:
        """Build one segmented schedule from the prepared prompt length.

        The full prompt inputs can be resident before its KV cache is complete,
        so request planning keeps prompt length and post-chunk cache length as
        separate terms. No memory calculation is needed while chunks execute.
        """
        if self.profile is None or not self.prefill_step_sizes:
            return None
        if prompt_context_length <= 0:
            raise ValueError("Prompt context length must be positive")
        if prompt_context_length > self.context_length:
            raise ValueError(
                f"Prepared prompt has {prompt_context_length:,} tokens, exceeding "
                f"the fitted {self.context_length:,}-token context"
            )

        segments: list[PrefillSegment] = []
        previous_end = 0
        for step_size in self.prefill_step_sizes:
            context_limit = _request_cache_limit_for_step(
                self.profile,
                step_size=step_size,
                prompt_context_length=prompt_context_length,
                safe_ceiling_bytes=self.safe_ceiling_bytes,
                baseline_bytes=self.baseline_bytes,
            )
            segment_end = min(prompt_context_length, context_limit)
            segment_end = max(previous_end, segment_end)
            if segment_end > previous_end:
                segments.append(
                    PrefillSegment(
                        end_context_length=segment_end,
                        step_size=step_size,
                    )
                )
                previous_end = segment_end
            if previous_end >= prompt_context_length:
                break

        if previous_end < prompt_context_length:
            # The reported maximum is constrained by the smallest candidate, so
            # this is only reachable through allocation-boundary rounding.
            segments.append(
                PrefillSegment(
                    end_context_length=prompt_context_length,
                    step_size=self.prefill_step_sizes[-1],
                )
            )

        return PrefillPlan(
            prompt_context_length=prompt_context_length,
            segments=tuple(segments),
        )


def fit_batched_vlm_context(
    *,
    model: Any,
    prefill_step_size: int,
) -> int | None:
    """Return the fitted token limit, preserving the existing integer API."""
    result = _fit_batched_vlm_context(
        model=model,
        prefill_step_size=prefill_step_size,
        dynamic_prefill=False,
    )
    return None if result is None else result.context_length


def fit_batched_vlm_context_result(
    *,
    model: Any,
    prefill_step_size: int,
) -> ContextFitResult | None:
    """Probe a model and return its context plus request-planning coefficients."""
    return _fit_batched_vlm_context(
        model=model,
        prefill_step_size=prefill_step_size,
        dynamic_prefill=True,
    )


def _fit_batched_vlm_context(
    *,
    model: Any,
    prefill_step_size: int,
    dynamic_prefill: bool,
) -> ContextFitResult | None:
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

        materializes_attention_scores = True
        if family == _FAMILY_GPT_OSS:
            query_head_dim = None
            for source in config_sources:
                query_head_dim = _config_value(source, "head_dim")
                if query_head_dim is not None:
                    break
            materializes_attention_scores = query_head_dim != 64
            if not materializes_attention_scores:
                validated_family = True
            else:
                logger.info(
                    "GPT-OSS head_dim=%s is not the validated fused layout; "
                    "retaining the conservative attention-score estimate",
                    query_head_dim,
                )

        query_attention_heads = None
        for key in ("num_attention_heads", "n_heads"):
            for source in config_sources:
                query_attention_heads = _config_value(source, key)
                if query_attention_heads is not None:
                    break
            if query_attention_heads is not None:
                break
        if query_attention_heads is None:
            logger.error(
                "Model context auto-fit could not find the query attention head count; "
                "leaving context unchanged"
            )
            return None

        try:
            profile = _probe_cache_fit_profile(
                model=model,
                language_model=language_model,
                family=family,
                max_context_length=max_context_length,
                query_attention_heads=query_attention_heads,
                prefill_step_size=prefill_step_size,
                materializes_attention_scores=materializes_attention_scores,
            )
        finally:
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
            mx.synchronize()

        if profile is None:
            return None

        baseline_bytes = mx.get_active_memory() + mx.get_cache_memory()
        working_set_bytes = mx.device_info()["max_recommended_working_set_size"]
        if dynamic_prefill:
            result = calculate_dynamic_context_fit(
                profile,
                working_set_bytes=working_set_bytes,
                baseline_bytes=baseline_bytes,
                maximum_prefill_step_size=prefill_step_size,
            )
        else:
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

        logger.info(
            "Model context auto-fit: family=%s max=%s fitted=%s "
            "working_set=%.2fGiB reserve=%.2fGiB baseline=%.2fGiB "
            "full_kv=%dB/token prompt_inputs=%dB/token "
            "attention_coefficient=%dB/context/step rotating_constant=%.2fGiB "
            "rotating_per_step=%dB steps=%s",
            profile.family,
            f"{max_context_length:,}",
            f"{result.context_length:,}",
            working_set_bytes / GIB,
            result.runtime_reserve_bytes / GIB,
            baseline_bytes / GIB,
            profile.full_kv_bytes_per_token,
            profile.prompt_input_bytes_per_token,
            profile.attention_bytes_per_context_per_prefill_token,
            profile.rotating_constant_bytes / GIB,
            profile.rotating_bytes_per_prefill_token,
            ",".join(f"{step}:{limit:,}" for step, limit in result.step_context_limits),
        )
        return result
    except Exception:
        logger.exception("Model context auto-fit failed; leaving context unchanged")
        return None


def _candidate_prefill_steps(maximum_prefill_step_size: int) -> tuple[int, ...]:
    if maximum_prefill_step_size <= 0:
        raise ValueError("Prefill step size must be positive")
    candidates = [maximum_prefill_step_size]
    candidates.extend(
        step for step in PREFILL_STEP_CANDIDATES if step < maximum_prefill_step_size
    )
    return tuple(dict.fromkeys(candidates))


def _attention_scores_bytes_per_token(
    profile: CacheFitProfile,
    prefill_step_size: int | None = None,
) -> int:
    if prefill_step_size is None:
        prefill_step_size = profile.prefill_step_size
    return profile.attention_bytes_per_context_per_prefill_token * prefill_step_size


def _context_fit_for_step(
    profile: CacheFitProfile,
    *,
    step_size: int,
    safe_ceiling_bytes: int,
    baseline_bytes: int,
) -> int:
    fixed_memory_bytes = (
        baseline_bytes
        + profile.fixed_ssm_bytes
        + profile.rotating_peak_bytes_for_step(step_size)
    )
    peak_bytes_per_token = (
        profile.full_kv_bytes_per_token
        + profile.prompt_input_bytes_per_token
        + _attention_scores_bytes_per_token(profile, step_size)
    )
    available_prompt_bytes = max(0, safe_ceiling_bytes - fixed_memory_bytes)
    tokens_that_fit = available_prompt_bytes // peak_bytes_per_token
    allocation_step = profile.allocation_step
    context_length = (
        (tokens_that_fit + allocation_step - 1) // allocation_step * allocation_step
    )
    return min(profile.max_context_length, context_length)


def _request_cache_limit_for_step(
    profile: CacheFitProfile,
    *,
    step_size: int,
    prompt_context_length: int,
    safe_ceiling_bytes: int,
    baseline_bytes: int,
) -> int:
    fixed_and_prompt_bytes = (
        baseline_bytes
        + profile.fixed_ssm_bytes
        + profile.rotating_peak_bytes_for_step(step_size)
        + profile.prompt_input_bytes_per_token * prompt_context_length
    )
    bytes_per_cached_token = (
        profile.full_kv_bytes_per_token
        + _attention_scores_bytes_per_token(profile, step_size)
    )
    available_cache_bytes = max(0, safe_ceiling_bytes - fixed_and_prompt_bytes)
    tokens_that_fit = available_cache_bytes // bytes_per_cached_token
    return tokens_that_fit // profile.allocation_step * profile.allocation_step


def _tail_limited_context(
    profile: CacheFitProfile,
    *,
    step_size: int,
    safe_ceiling_bytes: int,
    baseline_bytes: int,
) -> int:
    """Cap context so ``step_size`` stays safe through the fast-step region."""
    fixed_memory_bytes = (
        baseline_bytes
        + profile.fixed_ssm_bytes
        + profile.rotating_peak_bytes_for_step(step_size)
    )
    available_bytes = max(0, safe_ceiling_bytes - fixed_memory_bytes)
    cached_bytes_per_token = (
        profile.full_kv_bytes_per_token
        + _attention_scores_bytes_per_token(profile, step_size)
    )
    denominator = (
        profile.prompt_input_bytes_per_token * 100
        + cached_bytes_per_token * SMALLEST_STEP_START_CONTEXT_PERCENT
    )
    tokens_that_fit = available_bytes * 100 // denominator
    return tokens_that_fit // profile.allocation_step * profile.allocation_step


def calculate_dynamic_context_fit(
    profile: CacheFitProfile,
    *,
    working_set_bytes: int,
    baseline_bytes: int,
    maximum_prefill_step_size: int,
) -> ContextFitResult:
    """Fit context using a decreasing per-request prefill schedule."""
    # The measured overhead is tied to model execution, not installed RAM.
    runtime_reserve_bytes = MIN_RUNTIME_RESERVE_BYTES
    safe_ceiling_bytes = working_set_bytes - runtime_reserve_bytes
    prefill_step_sizes = _candidate_prefill_steps(maximum_prefill_step_size)
    step_context_limits = tuple(
        (
            step_size,
            _context_fit_for_step(
                profile,
                step_size=step_size,
                safe_ceiling_bytes=safe_ceiling_bytes,
                baseline_bytes=baseline_bytes,
            ),
        )
        for step_size in prefill_step_sizes
    )

    context_length = step_context_limits[-1][1]
    if len(prefill_step_sizes) >= 2 and prefill_step_sizes[-1] == 512:
        # Keep the slowest candidate in the final 15% regardless of the
        # caller's configured maximum (for example, 1,536 or 1,024).
        context_length = min(
            context_length,
            _tail_limited_context(
                profile,
                step_size=prefill_step_sizes[-2],
                safe_ceiling_bytes=safe_ceiling_bytes,
                baseline_bytes=baseline_bytes,
            ),
        )
    if context_length < MIN_FITTED_CONTEXT_TOKENS:
        logger.warning(
            "Model context auto-fit calculated %s tokens; using the %s token minimum",
            f"{context_length:,}",
            f"{MIN_FITTED_CONTEXT_TOKENS:,}",
        )
        context_length = MIN_FITTED_CONTEXT_TOKENS
    context_length = min(profile.max_context_length, context_length)

    return ContextFitResult(
        context_length=context_length,
        runtime_reserve_bytes=runtime_reserve_bytes,
        safe_ceiling_bytes=safe_ceiling_bytes,
        profile=profile,
        baseline_bytes=baseline_bytes,
        working_set_bytes=working_set_bytes,
        prefill_step_sizes=prefill_step_sizes,
        step_context_limits=step_context_limits,
    )


def calculate_context_fit(
    profile: CacheFitProfile,
    *,
    working_set_bytes: int,
    baseline_bytes: int,
) -> ContextFitResult:
    """Calculate the legacy fixed-step context limit."""
    runtime_reserve_bytes = MIN_RUNTIME_RESERVE_BYTES
    safe_ceiling_bytes = working_set_bytes - runtime_reserve_bytes
    context_length = _context_fit_for_step(
        profile,
        step_size=profile.prefill_step_size,
        safe_ceiling_bytes=safe_ceiling_bytes,
        baseline_bytes=baseline_bytes,
    )
    if context_length < MIN_FITTED_CONTEXT_TOKENS:
        logger.warning(
            "Model context auto-fit calculated %s tokens; using the %s token minimum",
            f"{context_length:,}",
            f"{MIN_FITTED_CONTEXT_TOKENS:,}",
        )
        context_length = MIN_FITTED_CONTEXT_TOKENS
    context_length = min(profile.max_context_length, context_length)
    return ContextFitResult(
        context_length=context_length,
        runtime_reserve_bytes=runtime_reserve_bytes,
        safe_ceiling_bytes=safe_ceiling_bytes,
        profile=profile,
        baseline_bytes=baseline_bytes,
        working_set_bytes=working_set_bytes,
        prefill_step_sizes=(profile.prefill_step_size,),
        step_context_limits=((profile.prefill_step_size, context_length),),
    )


def _probe_cache_fit_profile(
    *,
    model: Any,
    language_model: Any,
    family: str,
    max_context_length: int,
    query_attention_heads: int,
    prefill_step_size: int,
    materializes_attention_scores: bool = True,
) -> CacheFitProfile | None:
    prompt_cache = make_prompt_cache(language_model)
    input_ids = mx.zeros((1, 1), dtype=mx.int32)
    embedding_kwargs = {
        key: value
        for key, value in model.get_input_embeddings(input_ids).to_dict().items()
        if value is not None
    }
    inputs_embeds = embedding_kwargs.pop("inputs_embeds")
    prompt_input_bytes_per_token = inputs_embeds.nbytes
    per_layer_inputs = embedding_kwargs.get("per_layer_inputs")
    if per_layer_inputs is not None:
        prompt_input_bytes_per_token += per_layer_inputs.nbytes
    activation_dtype_bytes = inputs_embeds.itemsize

    language_model(
        input_ids,
        cache=prompt_cache,
        inputs_embeds=inputs_embeds,
        **embedding_kwargs,
    )
    mx.eval([cache.state for cache in prompt_cache])
    mx.synchronize()

    full_kv_bytes_per_token = 0
    rotating_constant_bytes = 0
    rotating_bytes_per_prefill_token = 0
    fixed_ssm_bytes = 0
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
            rotating_bytes_per_token = cache.nbytes // cache.keys.shape[2]
            rotating_constant_bytes += rotating_bytes_per_token * (cache.max_size - 1)
            rotating_bytes_per_prefill_token += rotating_bytes_per_token
            cache_allocation_steps.add(cache.step)
        else:
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
        prompt_input_bytes_per_token=prompt_input_bytes_per_token,
        materializes_attention_scores=materializes_attention_scores,
        query_attention_heads=query_attention_heads,
        activation_dtype_bytes=activation_dtype_bytes,
        prefill_step_size=prefill_step_size,
        rotating_constant_bytes=rotating_constant_bytes,
        rotating_bytes_per_prefill_token=rotating_bytes_per_prefill_token,
        fixed_ssm_bytes=fixed_ssm_bytes,
        max_context_length=max_context_length,
    )
