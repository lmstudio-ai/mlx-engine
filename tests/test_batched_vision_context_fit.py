from types import SimpleNamespace

import pytest

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from mlx_engine.model_kit.batched_vision import context_fit
from mlx_engine.model_kit.batched_vision.context_fit import (
    CacheFitProfile,
    calculate_context_fit,
    fit_batched_vlm_context,
)


GIB = 1024**3
MIB = 1024**2
KIB = 1024


class _EmbeddingOutput:
    def to_dict(self):
        return {"inputs_embeds": mx.ones((1, 1, 2), dtype=mx.float16)}


class _ProbeModel:
    def get_input_embeddings(self, _input_ids):
        return _EmbeddingOutput()


class _ProbeLanguageModel:
    config = SimpleNamespace(num_attention_heads=8)

    def __call__(self, *_args, **_kwargs):
        return None


def _initialized_kv_cache(cache=None, *, heads=1, head_dim=2, dtype=mx.float16):
    if cache is None:
        cache = KVCache()
    keys = mx.ones((1, heads, 1, head_dim), dtype=dtype)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    return cache


def _probe_profile(
    monkeypatch,
    prompt_cache,
    *,
    family="gemma4",
    language_model=None,
):
    if language_model is None:
        language_model = _ProbeLanguageModel()
    monkeypatch.setattr(context_fit, "make_prompt_cache", lambda _model: prompt_cache)
    return context_fit._probe_cache_fit_profile(
        model=_ProbeModel(),
        language_model=language_model,
        family=family,
        max_context_length=8_192,
        prefill_step_size=2_048,
    )


def _profile(
    *,
    family="gemma4",
    max_context_length=100_000,
    full_kv_bytes_per_token=MIB,
    prompt_embedding_bytes_per_token=4 * KIB,
    query_attention_heads=8,
    activation_dtype_bytes=2,
    prefill_step_size=2_048,
    rotating_peak_bytes=0,
    fixed_ssm_bytes=0,
):
    return CacheFitProfile(
        family=family,
        allocation_step=256,
        full_kv_bytes_per_token=full_kv_bytes_per_token,
        prompt_embedding_bytes_per_token=prompt_embedding_bytes_per_token,
        query_attention_heads=query_attention_heads,
        activation_dtype_bytes=activation_dtype_bytes,
        prefill_step_size=prefill_step_size,
        rotating_peak_bytes=rotating_peak_bytes,
        fixed_ssm_bytes=fixed_ssm_bytes,
        max_context_length=max_context_length,
    )


def _peak_bytes_per_token(profile):
    return (
        profile.full_kv_bytes_per_token
        + profile.prompt_embedding_bytes_per_token
        + profile.query_attention_heads
        * profile.prefill_step_size
        * profile.activation_dtype_bytes
    )


def test_probe_sums_full_kv_bytes_from_real_allocation(monkeypatch):
    first_cache = _initialized_kv_cache(heads=1, head_dim=2)
    second_cache = _initialized_kv_cache(heads=2, head_dim=4)

    profile = _probe_profile(monkeypatch, [first_cache, second_cache])

    assert profile.allocation_step == 256
    assert profile.full_kv_bytes_per_token == (
        first_cache.nbytes // 256 + second_cache.nbytes // 256
    )
    assert profile.prompt_embedding_bytes_per_token == 4
    assert profile.query_attention_heads == 8
    assert profile.activation_dtype_bytes == 2
    assert profile.prefill_step_size == 2_048


def test_probe_uses_rotating_prefill_peak(monkeypatch):
    kv_cache = _initialized_kv_cache()
    rotating_cache = _initialized_kv_cache(RotatingKVCache(max_size=1_024, keep=0))

    profile = _probe_profile(monkeypatch, [kv_cache, rotating_cache])

    rotating_bytes_per_token = rotating_cache.nbytes // rotating_cache.keys.shape[2]
    assert profile.rotating_peak_bytes == rotating_bytes_per_token * (1_024 + 2_048 - 1)


def test_probe_counts_actual_qwen_arrays_state_bytes(monkeypatch):
    arrays_cache = ArraysCache(size=2)
    arrays_cache[0] = mx.ones((1, 3, 4), dtype=mx.float16)
    arrays_cache[1] = mx.ones((1, 2, 2), dtype=mx.float16)
    mx.eval(arrays_cache.state)
    language_model = _ProbeLanguageModel()
    language_model.config = SimpleNamespace()
    language_model.args = SimpleNamespace(num_attention_heads=16)

    profile = _probe_profile(
        monkeypatch,
        [_initialized_kv_cache(), arrays_cache],
        family="qwen3_5",
        language_model=language_model,
    )

    assert profile.fixed_ssm_bytes == arrays_cache.nbytes
    assert profile.query_attention_heads == 16


def test_probe_rejects_unknown_cache_type(monkeypatch, caplog):
    class UnknownCache:
        state = []

    profile = _probe_profile(
        monkeypatch,
        [_initialized_kv_cache(), UnknownCache()],
    )

    assert profile is None
    assert "Unsupported UnknownCache" in caplog.text


def test_probe_rejects_rotating_keep(monkeypatch):
    rotating_cache = _initialized_kv_cache(RotatingKVCache(max_size=1_024, keep=4))

    profile = _probe_profile(
        monkeypatch,
        [_initialized_kv_cache(), rotating_cache],
    )

    assert profile is None


def test_probe_rejects_inconsistent_allocation_steps(monkeypatch):
    alternate_cache = type("KVCache", (), {})()
    alternate_cache.keys = mx.ones((1, 1, 128, 1), dtype=mx.float16)
    alternate_cache.nbytes = alternate_cache.keys.nbytes * 2
    alternate_cache.step = 128
    alternate_cache.state = (alternate_cache.keys, alternate_cache.keys)

    profile = _probe_profile(
        monkeypatch,
        [_initialized_kv_cache(), alternate_cache],
    )

    assert profile is None


def test_probe_rejects_empty_qwen_state(monkeypatch):
    arrays_cache = ArraysCache(size=2)
    arrays_cache[0] = mx.ones((1, 3, 4), dtype=mx.float16)

    profile = _probe_profile(
        monkeypatch,
        [_initialized_kv_cache(), arrays_cache],
        family="qwen3_5",
    )

    assert profile is None


def test_fit_clamps_to_max_context():
    result = calculate_context_fit(
        _profile(
            max_context_length=8_192,
            full_kv_bytes_per_token=512 * KIB,
        ),
        working_set_bytes=10 * GIB,
        baseline_bytes=0,
    )

    assert result.context_length == 8_192
    assert result.safe_ceiling_bytes == 7 * GIB


def test_fit_rounds_up_to_allocation_step():
    profile = _profile()
    result = calculate_context_fit(
        profile,
        working_set_bytes=10 * GIB,
        baseline_bytes=7 * GIB - 5_000 * _peak_bytes_per_token(profile),
    )

    assert result.context_length == 5_120


def test_fit_accepts_exactly_minimum_context():
    profile = _profile()
    result = calculate_context_fit(
        profile,
        working_set_bytes=10 * GIB,
        baseline_bytes=7 * GIB - 4_096 * _peak_bytes_per_token(profile),
    )

    assert result.context_length == 4_096


def test_fit_raises_a_small_result_to_minimum(caplog):
    profile = _profile()
    result = calculate_context_fit(
        profile,
        working_set_bytes=10 * GIB,
        baseline_bytes=7 * GIB - 2_000 * _peak_bytes_per_token(profile),
    )

    assert result.context_length == 4_096
    assert "using the 4,096 token minimum" in caplog.text


def test_runtime_reserve_uses_five_percent_or_three_gib():
    percentage_result = calculate_context_fit(
        _profile(max_context_length=4_096),
        working_set_bytes=80 * GIB,
        baseline_bytes=0,
    )
    minimum_result = calculate_context_fit(
        _profile(max_context_length=4_096),
        working_set_bytes=16 * GIB,
        baseline_bytes=0,
    )

    assert percentage_result.runtime_reserve_bytes == 4 * GIB
    assert percentage_result.safe_ceiling_bytes == 76 * GIB
    assert minimum_result.runtime_reserve_bytes == 3 * GIB
    assert minimum_result.safe_ceiling_bytes == 13 * GIB


def test_fit_uses_minimum_when_fixed_memory_reaches_ceiling():
    result = calculate_context_fit(
        _profile(fixed_ssm_bytes=GIB),
        working_set_bytes=10 * GIB,
        baseline_bytes=6 * GIB,
    )

    assert result.context_length == 4_096


# These profiles use measurements from the target models on a 27 GiB working set.
@pytest.mark.parametrize(
    ("profile", "baseline_bytes", "expected_context"),
    [
        pytest.param(
            _profile(
                family="qwen3_5",
                max_context_length=262_144,
                full_kv_bytes_per_token=12_288,
                prompt_embedding_bytes_per_token=4_096,
                query_attention_heads=8,
                fixed_ssm_bytes=19_537_920,
            ),
            1_722_151_626,
            262_144,
            id="qwen-3.5-2b-dense-4bit",
        ),
        pytest.param(
            _profile(
                family="qwen3_5",
                max_context_length=262_144,
                full_kv_bytes_per_token=20_480,
                prompt_embedding_bytes_per_token=4_096,
                query_attention_heads=16,
                fixed_ssm_bytes=64_389_120,
            ),
            20_402_597_098,
            58_880,
            id="qwen-3.6-35b-a3b-moe-4bit",
        ),
        pytest.param(
            _profile(
                family="qwen3_5",
                max_context_length=262_144,
                full_kv_bytes_per_token=65_536,
                prompt_embedding_bytes_per_token=10_240,
                query_attention_heads=24,
                fixed_ssm_bytes=153_944_064,
            ),
            16_055_717_354,
            55_040,
            id="qwen-3.6-27b-dense-4bit",
        ),
        pytest.param(
            _profile(
                max_context_length=131_072,
                full_kv_bytes_per_token=6_144,
                prompt_embedding_bytes_per_token=3_072,
                query_attention_heads=8,
                rotating_peak_bytes=31_444_992,
            ),
            4_339_854_220,
            131_072,
            id="gemma-4-e2b-dense-4bit",
        ),
        pytest.param(
            _profile(
                max_context_length=262_144,
                full_kv_bytes_per_token=20_480,
                prompt_embedding_bytes_per_token=5_632,
                query_attention_heads=16,
                rotating_peak_bytes=628_940_800,
            ),
            15_611_655_610,
            104_192,
            id="gemma-4-26b-a4b-moe-4bit",
        ),
        pytest.param(
            _profile(
                max_context_length=262_144,
                full_kv_bytes_per_token=16_384,
                prompt_embedding_bytes_per_token=7_680,
                query_attention_heads=16,
                rotating_peak_bytes=1_006_305_280,
            ),
            10_990_079_092,
            153_856,
            id="gemma-4-12b-dense-4bit",
        ),
        pytest.param(
            _profile(
                max_context_length=262_144,
                full_kv_bytes_per_token=16_384,
                prompt_embedding_bytes_per_token=7_680,
                query_attention_heads=16,
                rotating_peak_bytes=1_006_305_280,
            ),
            12_718_509_172,
            134_656,
            id="gemma-4-12b-dense-8bit",
        ),
    ],
)
def test_fit_matches_measured_model_profiles(profile, baseline_bytes, expected_context):
    result = calculate_context_fit(
        profile,
        working_set_bytes=27 * GIB,
        baseline_bytes=baseline_bytes,
    )

    assert result.context_length == expected_context


@pytest.mark.parametrize(
    ("working_set_gib", "expected_context"),
    [
        pytest.param(16, 4_096, id="lower-memory"),
        pytest.param(27, 134_656, id="tested-memory"),
        pytest.param(48, 262_144, id="higher-memory"),
        pytest.param(96, 262_144, id="percentage-reserve"),
    ],
)
def test_fit_scales_across_memory_tiers(working_set_gib, expected_context):
    profile = _profile(
        max_context_length=262_144,
        full_kv_bytes_per_token=16_384,
        prompt_embedding_bytes_per_token=7_680,
        query_attention_heads=16,
        rotating_peak_bytes=1_006_305_280,
    )

    result = calculate_context_fit(
        profile,
        working_set_bytes=working_set_gib * GIB,
        baseline_bytes=12_718_509_172,
    )

    assert result.context_length == expected_context


def test_one_token_probe_uses_token_zero_and_reports_fit(monkeypatch):
    prompt_cache = [KVCache()]
    calls = []

    class LanguageModel:
        model_type = "gemma4_text"
        config = SimpleNamespace(
            max_position_embeddings=8_192,
            num_attention_heads=1,
        )

        def __call__(self, input_ids, *, cache, inputs_embeds, **kwargs):
            calls.append((input_ids.tolist(), inputs_embeds.shape, kwargs))
            keys = mx.ones((1, 1, 1, 2), dtype=mx.float16)
            cache[0].update_and_fetch(keys, keys)

    language_model = LanguageModel()
    model = SimpleNamespace(
        language_model=language_model,
        get_input_embeddings=lambda _input_ids: _EmbeddingOutput(),
    )
    monkeypatch.setattr(context_fit, "make_prompt_cache", lambda _model: prompt_cache)
    monkeypatch.setattr(context_fit.mx, "get_active_memory", lambda: GIB)
    monkeypatch.setattr(context_fit.mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(
        context_fit.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 10 * GIB},
    )

    context_length = fit_batched_vlm_context(
        model=model,
        prefill_step_size=2_048,
    )

    assert calls == [([[0]], (1, 1, 2), {})]
    assert context_length == 8_192


def test_qwen3_6_uses_args_context(monkeypatch):
    language_model = SimpleNamespace(
        model_type="qwen3_6_moe",
        config=SimpleNamespace(),
        args=SimpleNamespace(max_position_embeddings=8_192),
    )
    model = SimpleNamespace(language_model=language_model)
    monkeypatch.setattr(
        context_fit,
        "_probe_cache_fit_profile",
        lambda **_kwargs: _profile(
            max_context_length=8_192,
            full_kv_bytes_per_token=512 * KIB,
        ),
    )
    monkeypatch.setattr(context_fit.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(context_fit.mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(
        context_fit.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 10 * GIB},
    )

    assert fit_batched_vlm_context(model=model, prefill_step_size=2_048) == 8_192


def test_fit_logs_and_falls_back_for_unsupported_family(caplog):
    language_model = SimpleNamespace(
        model_type="unsupported",
        config=SimpleNamespace(max_position_embeddings=16_384),
    )
    model = SimpleNamespace(language_model=language_model)

    assert fit_batched_vlm_context(model=model, prefill_step_size=2_048) == 16_384
    assert "using max context 16,384" in caplog.text
