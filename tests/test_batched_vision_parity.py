"""High-level batched VLM parity tests."""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import mlx.core as mx
import mlx_vlm
import pytest
from mlx_vlm.generate import generate_step as vlm_generate_step

from mlx_engine.generate import load_model, tokenize, unload
from mlx_engine.model_kit.batched_vision import BatchedVisionModelKit
from mlx_engine.model_kit.batched_vision.prompt_inputs import get_image_token_index
from mlx_engine.utils.image_utils import convert_to_pil

from tests.patched_model_test_utils import get_real_model_path, max_abs_diff
from tests.shared import RecordingReporter, read_image_b64

pytestmark = pytest.mark.heavy


class _CaptureLogitsProcessor:
    def __init__(self):
        self.logits = []

    def __call__(self, _tokens, logits):
        captured = mx.array(logits)
        mx.eval(captured)
        self.logits.append(captured)
        return logits


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


@dataclass(frozen=True)
class VlmParityCase:
    id: str
    model_name: str
    text_prompt: Callable[[Path, object | None], str]
    image_prompt: Callable[[Path, object | None], str]
    restore_prompt: Callable[[Path, object | None], str]
    restore_cached_tokens: int = 2048


def _qwen_text_prompt(_model_path: Path, _processor: object | None) -> str:
    return (
        "<|im_start|>user\n"
        "Tell me one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _qwen_image_prompt(_model_path: Path, _processor: object | None) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "What is in the image?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _qwen_restore_prompt(_model_path: Path, _processor: object | None) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        + ("Remember the word meridian. " * 330)
        + "What is in the image?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _gemma4_chat_prompt(
    _model_path: Path,
    processor: object | None,
    prompt: str,
    *,
    image: bool,
) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Gemma4 parity prompts require a loaded processor.")
    content = []
    if image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _gemma4_text_prompt(model_path: Path, processor: object | None) -> str:
    return _gemma4_chat_prompt(
        model_path,
        processor,
        "Tell me one short sentence.",
        image=False,
    )


def _gemma4_image_prompt(model_path: Path, processor: object | None) -> str:
    return _gemma4_chat_prompt(
        model_path,
        processor,
        "What is in the image?",
        image=True,
    )


def _gemma4_restore_prompt(model_path: Path, processor: object | None) -> str:
    return _gemma4_chat_prompt(
        model_path,
        processor,
        ("Remember the word meridian. " * 360) + "What is in the image?",
        image=True,
    )


VLM_PARITY_CASES = [
    pytest.param(
        VlmParityCase(
            id="qwen3_5",
            model_name="lmstudio-community/Qwen3.5-2B-MLX-4bit",
            text_prompt=_qwen_text_prompt,
            image_prompt=_qwen_image_prompt,
            restore_prompt=_qwen_restore_prompt,
        ),
        id="qwen3_5",
    ),
    pytest.param(
        VlmParityCase(
            id="gemma4",
            model_name="lmstudio-community/gemma-4-E2B-it-MLX-4bit",
            text_prompt=_gemma4_text_prompt,
            image_prompt=_gemma4_image_prompt,
            restore_prompt=_gemma4_restore_prompt,
        ),
        id="gemma4",
    ),
]

GEMMA4_12B_PARITY_CASE = VlmParityCase(
    id="gemma4_12b",
    model_name="lmstudio-community/gemma-4-12B-it-MLX-4bit",
    text_prompt=_gemma4_text_prompt,
    image_prompt=_gemma4_image_prompt,
    restore_prompt=_gemma4_restore_prompt,
)

VLM_GENERATION_TRACE_CASES = VLM_PARITY_CASES + [
    pytest.param(GEMMA4_12B_PARITY_CASE, id="gemma4_12b"),
]


def _toucan_b64() -> str:
    return read_image_b64(Path(__file__).parent.parent / "demo-data" / "toucan.jpeg")


def _wait_for_disk_cache_records(model_kit: BatchedVisionModelKit) -> None:
    for _ in range(100):
        stats = model_kit._prompt_cache_store.snapshot_stats()
        if stats.entry_count > 0 and any(stats.chunk_records_available_by_key.values()):
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for VLM prompt cache records.")


def _trace_responses(responses):
    return [(response.token, response.token_logprob) for response in responses]


def _assert_token_trace_matches(actual, expected, *, logprob_abs_tol: float) -> None:
    assert [token for token, _ in actual] == [token for token, _ in expected]
    assert [logprob for _, logprob in actual] == pytest.approx(
        [logprob for _, logprob in expected],
        abs=logprob_abs_tol,
    )


@pytest.mark.parametrize("case", VLM_PARITY_CASES)
def test_vlm_image_prompt_restore_matches_same_schedule(case: VlmParityCase):
    """High-level image restores must match a fresh cache with the same schedule."""
    model_path = get_real_model_path(case.model_name)
    image_b64 = _toucan_b64()
    model_kit = load_model(
        model_path=model_path,
        max_kv_size=4096,
        trust_remote_code=True,
    )
    assert isinstance(model_kit, BatchedVisionModelKit)
    prompt = case.restore_prompt(model_path, model_kit.processor)

    def run_request(request_id: str):
        processor = _CaptureLogitsProcessor()
        reporter = RecordingReporter()
        responses = list(
            model_kit.generate(
                prompt_tokens=tokenize(model_kit, prompt),
                request_id=request_id,
                images_b64=[image_b64],
                prompt_progress_reporter=reporter,
                top_logprobs=0,
                max_tokens=1,
                sampler=_greedy,
                logits_processors=[processor],
            )
        )
        assert len(responses) == 1
        # Prompt-final logits plus the first decode-ahead logits.
        assert len(processor.logits) == 2
        return processor.logits, reporter, responses[0]

    try:
        # The prompt is intentionally longer than the default VLM prefill step
        # size. The first request computes and saves the 2048-token prefix, then
        # computes the suffix. The second request restores that same prefix from
        # disk, then computes the same suffix. Comparing those two paths avoids
        # false failures from comparing one-shot prefill against restored-prefix
        # prefill, which is a different numerical schedule.
        first_logits, first_reporter, first_response = run_request("fresh")
        assert first_reporter.events[0]["cached_tokens"] == 0
        _wait_for_disk_cache_records(model_kit)

        restored_logits, restored_reporter, restored_response = run_request("restored")
        assert (
            restored_reporter.events[0]["cached_tokens"] == case.restore_cached_tokens
        )
        assert restored_response.token == first_response.token
        assert restored_response.token_logprob == first_response.token_logprob

        for actual, expected in zip(restored_logits, first_logits, strict=True):
            diff = max_abs_diff(actual, expected)
            assert diff == 0.0, (
                f"{case.id} image prompt restore changed same-schedule logits "
                f"(max diff {diff:.6f})."
            )
    finally:
        unload(model_kit)


@pytest.mark.parametrize("case", VLM_GENERATION_TRACE_CASES)
def test_vlm_generation_trace_matches_mlx_vlm_same_inputs(case: VlmParityCase):
    """Batched engine generation must match mlx-vlm's token/logit trace."""
    model_path = get_real_model_path(case.model_name)
    image_b64 = _toucan_b64()
    max_tokens = 2

    def run_upstream(model, processor, config, prompt: str, image: str | None):
        processor_capture = _CaptureLogitsProcessor()
        # Use the same prompt IDs / processor tensors that the batched path
        # should derive, so this catches orchestration drift instead of template
        # or image-loading differences.
        if image is None:
            input_ids = mx.array(
                processor.tokenizer.encode(prompt, add_special_tokens=False),
                dtype=mx.int32,
            )[None, :]
            pixel_values = None
            mask = None
            prompt_kwargs = {}
        else:
            raw_inputs = mlx_vlm.prepare_inputs(
                processor=processor,
                images=convert_to_pil([image]),
                prompts=prompt,
                image_token_index=get_image_token_index(config),
                resize_shape=None,
            )
            input_ids = raw_inputs["input_ids"]
            pixel_values = raw_inputs.get("pixel_values")
            mask = raw_inputs.get("attention_mask")
            prompt_kwargs = {
                key: value
                for key, value in raw_inputs.items()
                if key not in {"input_ids", "pixel_values", "attention_mask"}
            }

        tokens = []
        token_logprobs = []
        for token, logprobs in vlm_generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            max_tokens=max_tokens,
            sampler=_greedy,
            logits_processors=[processor_capture],
            **prompt_kwargs,
        ):
            mx.eval(logprobs)
            tokens.append(token)
            token_logprobs.append(float(logprobs[token].item()))
        return processor_capture.logits, tokens, token_logprobs

    def run_engine(name: str, prompt: str, image: str | None):
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=4096,
            trust_remote_code=True,
        )
        assert isinstance(model_kit, BatchedVisionModelKit)
        try:
            processor_capture = _CaptureLogitsProcessor()
            responses = list(
                model_kit.generate(
                    prompt_tokens=tokenize(model_kit, prompt),
                    request_id=name,
                    images_b64=None if image is None else [image],
                    prompt_progress_reporter=None,
                    top_logprobs=0,
                    max_tokens=max_tokens,
                    sampler=_greedy,
                    logits_processors=[processor_capture],
                )
            )
            return (
                processor_capture.logits,
                [response.token for response in responses],
                [response.token_logprob for response in responses],
            )
        finally:
            unload(model_kit)

    loaded = mlx_vlm.utils.load(model_path, trust_remote_code=True)
    if len(loaded) == 3:
        upstream_model, upstream_processor, config = loaded
    else:
        upstream_model, upstream_processor = loaded
        config = mlx_vlm.utils.load_config(model_path, trust_remote_code=True)
    prompt_cases = [
        ("text", case.text_prompt(model_path, upstream_processor), None),
        ("image", case.image_prompt(model_path, upstream_processor), image_b64),
    ]
    try:
        upstream_traces = {
            name: run_upstream(
                upstream_model, upstream_processor, config, prompt, image
            )
            for name, prompt, image in prompt_cases
        }
    finally:
        del upstream_model
        del upstream_processor
        gc.collect()
        mx.clear_cache()

    engine_traces = {
        name: run_engine(name, prompt, image) for name, prompt, image in prompt_cases
    }

    for name, _, _ in prompt_cases:
        upstream_logits, upstream_tokens, upstream_logprobs = upstream_traces[name]
        engine_logits, engine_tokens, engine_logprobs = engine_traces[name]

        assert engine_tokens == upstream_tokens
        assert len(engine_logits) == max_tokens + 1
        assert len(upstream_logits) == max_tokens + 1
        if case.id == "gemma4_12b" and name == "image":
            # Gemma4 unified image prompts intentionally split prefill at the
            # visual-prefix boundary so later text can use normal cache chunks
            # without splitting the bidirectional visual attention span. That
            # gives 12B a different BF16/quantized schedule than mlx-vlm's
            # direct one-shot reference; disabling only that split restores
            # exact logits. Keep this case focused on emitted tokens and
            # selected logprob stability.
            assert engine_logprobs == pytest.approx(upstream_logprobs, abs=0.125)
            continue

        assert engine_logprobs == upstream_logprobs
        for step, (engine_step_logits, upstream_step_logits) in enumerate(
            zip(engine_logits, upstream_logits, strict=True)
        ):
            diff = max_abs_diff(engine_step_logits, upstream_step_logits)
            assert diff == 0.0, (
                f"{case.id} {name} generation step {step} changed mlx-vlm logits "
                f"(max diff {diff:.6f})."
            )


@pytest.mark.parametrize("case", VLM_PARITY_CASES)
def test_vlm_continuous_batching_matches_independent_requests(case: VlmParityCase):
    """Mixed text/image batching must not change request-local token traces.

    The trace is the emitted token IDs plus each selected token's normalized
    logprob. We intentionally do not compare full logits here: co-resident
    text/image rows can use a different batched numerical path. Tokens must
    match exactly; selected logprobs get BF16 headroom for shape-dependent MLX
    SDPA drift.
    """
    model_path = get_real_model_path(case.model_name)
    image_b64 = _toucan_b64()
    max_tokens = 4
    prompt_specs = [
        ("text", case.text_prompt, None),
        ("image", case.image_prompt, image_b64),
    ]

    def run_request(model_kit, name: str, prompt: str, image: str | None):
        processor = _CaptureLogitsProcessor()
        responses = list(
            model_kit.generate(
                prompt_tokens=tokenize(model_kit, prompt),
                request_id=name,
                images_b64=None if image is None else [image],
                prompt_progress_reporter=None,
                top_logprobs=0,
                max_tokens=max_tokens,
                sampler=_greedy,
                logits_processors=[processor],
            )
        )
        assert len(processor.logits) == max_tokens + 1
        return _trace_responses(responses)

    independent_traces = {}
    for name, build_prompt, image in prompt_specs:
        model_kit = load_model(
            model_path=model_path,
            max_kv_size=4096,
            max_seq_nums=1,
            trust_remote_code=True,
        )
        assert isinstance(model_kit, BatchedVisionModelKit)
        try:
            prompt = build_prompt(model_path, model_kit.processor)
            independent_traces[name] = run_request(model_kit, name, prompt, image)
        finally:
            unload(model_kit)

    model_kit = load_model(
        model_path=model_path,
        max_kv_size=4096,
        max_seq_nums=2,
        trust_remote_code=True,
    )
    assert isinstance(model_kit, BatchedVisionModelKit)
    try:
        streams = []
        processors = {}
        for name, build_prompt, image in prompt_specs:
            prompt = build_prompt(model_path, model_kit.processor)
            processor = _CaptureLogitsProcessor()
            processors[name] = processor
            streams.append(
                (
                    name,
                    model_kit.generate(
                        prompt_tokens=tokenize(model_kit, prompt),
                        request_id=name,
                        images_b64=None if image is None else [image],
                        prompt_progress_reporter=None,
                        top_logprobs=0,
                        max_tokens=max_tokens,
                        sampler=_greedy,
                        logits_processors=[processor],
                    ),
                )
            )

        concurrent_traces = {}
        errors = []

        def consume(name: str, stream) -> None:
            try:
                concurrent_traces[name] = _trace_responses(list(stream))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=consume, args=(name, stream))
            for name, stream in streams
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
    finally:
        unload(model_kit)

    for name, _, _ in prompt_specs:
        _assert_token_trace_matches(
            concurrent_traces[name],
            independent_traces[name],
            logprob_abs_tol=0.125,
        )
        assert len(processors[name].logits) == max_tokens + 1
