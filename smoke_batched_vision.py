import argparse
import base64
import os
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mlx.core as mx
from transformers import AutoProcessor

from mlx_engine.generate import create_generator, load_model, tokenize, unload
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    RestoredPromptCache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.chunks import (
    build_prefix_cache_chunks,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.image_spans import (
    image_safe_common_prefix_len,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptImageSpan,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    make_record_key,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import (
    PromptCacheRestorePlan,
    VlmPromptCacheStore,
)
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    prepare_prompt_inputs as prepare_batched_vision_prompt_inputs,
)
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

DEFAULT_MODEL_PATH = Path(
    "~/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-MLX-4bit"
).expanduser()
DEFAULT_QWEN35_MODEL_PATH = Path(
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"
).expanduser()
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "demo-data" / "toucan.jpeg"
DEFAULT_SECONDARY_IMAGE_PATH = (
    Path(__file__).resolve().parent / "demo-data" / "chameleon.webp"
)
DEFAULT_PROMPT = "Describe this image in one short sentence."


@dataclass(frozen=True)
class RequestSpec:
    name: str
    prompt: str
    expected_any: tuple[str, ...]
    image_b64: str | None = None
    max_tokens: int = 16
    top_logprobs: int = 0
    start_delay_s: float = 0.0


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    name: str
    prompt_tokens: list[int]
    images_b64: list[str] | None
    expected_any: tuple[str, ...]
    max_tokens: int
    top_logprobs: int
    start_delay_s: float


@dataclass
class RequestResult:
    text: str
    stop_reason: str | None
    top_logprobs_counts: list[int]
    elapsed_s: float


@dataclass
class PromptProgressTrace:
    begin_cached_tokens: int | None = None
    begin_total_prompt_tokens: int | None = None
    begin_prefill_tokens_processed: int | None = None
    updates: list[int] = field(default_factory=list)
    finish_prefill_tokens_processed: int | None = None


class TracePromptProgressReporter(PromptProgressReporter):
    def __init__(self, trace: PromptProgressTrace):
        self._trace = trace

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        if not is_draft and self._trace.begin_prefill_tokens_processed is None:
            self._trace.begin_cached_tokens = cached_tokens
            self._trace.begin_total_prompt_tokens = total_prompt_tokens
            self._trace.begin_prefill_tokens_processed = prefill_tokens_processed
        return True

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        if not is_draft:
            self._trace.updates.append(prefill_tokens_processed)
        return True

    def finish(
        self, is_draft: bool, prefill_tokens_processed: int | None = None
    ) -> bool:
        if not is_draft:
            self._trace.finish_prefill_tokens_processed = prefill_tokens_processed
        return True


OPTIONAL_RECORD_KINDS = {
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke and stress test the mlx-vlm batched vision path"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "stress",
            "prefix",
            "prefix-restart",
            "multi-prefix",
            "rotating-suffix",
            "multi-prefix-e2e",
            "hybrid-cache-e2e",
            "arrays-cache-e2e",
            "arrays-cache-branch-e2e",
            "qwen-rope-e2e",
            "eviction",
            "eviction-e2e",
            "eviction-stress",
            "cross-thread-restore",
        ],
        default="smoke",
        help="Which verification flow to run",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local path to the MLX model directory",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Primary local image for multimodal requests",
    )
    parser.add_argument(
        "--secondary-image",
        type=Path,
        default=DEFAULT_SECONDARY_IMAGE_PATH,
        help="Secondary local image for stress mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="User prompt to send with the primary image in smoke mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="Maximum number of generated tokens in smoke mode",
    )
    parser.add_argument(
        "--max-seq-nums",
        type=int,
        default=4,
        help="Parallel slots to configure on load_model",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default="smoke-mm-1",
        help="Request id passed into create_generator in smoke mode",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of stress rounds to execute",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=80,
        help="Inter-request arrival stagger used by stress mode",
    )
    parser.add_argument(
        "--join-timeout-s",
        type=float,
        default=60.0,
        help="Per-thread join timeout in stress mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic stress request ordering",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to model loading and processor loading",
    )
    parser.add_argument(
        "--cache-max-bytes",
        type=int,
        default=None,
        help="Optional batched-vision disk-cache byte budget",
    )
    parser.add_argument(
        "--prefix-wait-timeout-s",
        type=float,
        default=10.0,
        help="How long prefix mode waits for the async prompt snapshot save to land",
    )
    return parser.parse_args()


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def build_prompt(processor, prompt: str, image_b64: str | None) -> str:
    content = [{"type": "text", "text": prompt}]
    if image_b64 is not None:
        content.insert(0, {"type": "image", "base64": image_b64})
    conversation = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_request(
    model_kit,
    processor,
    spec: RequestSpec,
    request_id: str,
) -> PreparedRequest:
    prompt = build_prompt(processor, spec.prompt, spec.image_b64)
    return prepare_request_from_prompt_text(
        model_kit,
        prompt_text=prompt,
        spec=spec,
        request_id=request_id,
    )


def prepare_request_from_prompt_text(
    model_kit,
    *,
    prompt_text: str,
    spec: RequestSpec,
    request_id: str,
) -> PreparedRequest:
    return PreparedRequest(
        request_id=request_id,
        name=spec.name,
        prompt_tokens=tokenize(model_kit, prompt_text),
        images_b64=[spec.image_b64] if spec.image_b64 is not None else None,
        expected_any=spec.expected_any,
        max_tokens=spec.max_tokens,
        top_logprobs=spec.top_logprobs,
        start_delay_s=spec.start_delay_s,
    )


def run_prepared_request(
    model_kit,
    prepared: PreparedRequest,
    *,
    prompt_progress_reporter: PromptProgressReporter | None = None,
) -> RequestResult:
    if prepared.start_delay_s > 0:
        time.sleep(prepared.start_delay_s)

    start_time = time.perf_counter()
    text = ""
    stop_reason = None
    top_logprobs_counts: list[int] = []

    for result in create_generator(
        model_kit=model_kit,
        prompt_tokens=prepared.prompt_tokens,
        images_b64=prepared.images_b64,
        prompt_progress_reporter=prompt_progress_reporter,
        temp=0.0,
        max_tokens=prepared.max_tokens,
        top_logprobs=prepared.top_logprobs,
        request_id=prepared.request_id,
    ):
        text += result.text
        if result.top_logprobs is not None:
            top_logprobs_counts.append(len(result.top_logprobs))
        if result.stop_condition:
            stop_reason = result.stop_condition.stop_reason
            break

    return RequestResult(
        text=text,
        stop_reason=stop_reason,
        top_logprobs_counts=top_logprobs_counts,
        elapsed_s=time.perf_counter() - start_time,
    )


def truncate_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def validate_result(prepared: PreparedRequest, result: RequestResult) -> None:
    if len(result.text.strip()) == 0:
        raise RuntimeError(f"{prepared.request_id}: no text generated")
    if result.stop_reason is None:
        raise RuntimeError(f"{prepared.request_id}: missing stop reason")
    lowered = result.text.lower()
    if prepared.expected_any and not any(
        expected.lower() in lowered for expected in prepared.expected_any
    ):
        raise RuntimeError(
            f"{prepared.request_id}: output {result.text!r} did not include any of {prepared.expected_any!r}"
        )
    if prepared.top_logprobs > 0 and not result.top_logprobs_counts:
        raise RuntimeError(
            f"{prepared.request_id}: expected top_logprobs but none were returned"
        )


def prepare_prompt_inputs(
    model_kit,
    prepared: PreparedRequest,
):
    # Use the real mlx-vlm prompt-prep path so prefix validation checks the
    # actual disk-cache key space instead of guessing from raw text.
    return prepare_batched_vision_prompt_inputs(
        prompt_tokens=prepared.prompt_tokens,
        images_b64=prepared.images_b64,
        max_image_size=None,
        tokenizer=model_kit.tokenizer,
        processor=model_kit.processor,
        config=model_kit.config,
    )


def is_strict_prefix(prefix: list[int], values: list[int]) -> bool:
    return len(prefix) < len(values) and values[: len(prefix)] == prefix


def common_token_prefix_len(left: list[int], right: list[int]) -> int:
    for idx, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return idx
    return min(len(left), len(right))


def build_prefix_requests(
    model_kit,
    processor,
    *,
    prompt: str,
    primary_image_b64: str,
    max_tokens: int = 24,
) -> tuple[PreparedRequest, PreparedRequest, object, object, str]:
    base_prompt_text = build_prompt(processor, prompt, primary_image_b64)
    base_prepared = prepare_request_from_prompt_text(
        model_kit,
        prompt_text=base_prompt_text,
        spec=RequestSpec(
            name="prefix-base",
            prompt=prompt,
            expected_any=("bird", "toucan"),
            image_b64=primary_image_b64,
            max_tokens=max_tokens,
        ),
        request_id="prefix-base",
    )
    base_prompt_inputs = prepare_prompt_inputs(model_kit, base_prepared)

    for suffix in [
        "\nA",
        "\nThe",
        "\nIt",
        " .",
        " ...",
        "\n",
        "\n\n",
        "\n ",
    ]:
        extended_prepared = prepare_request_from_prompt_text(
            model_kit,
            prompt_text=base_prompt_text + suffix,
            spec=RequestSpec(
                name="prefix-extended",
                prompt=prompt,
                expected_any=(),
                image_b64=primary_image_b64,
                max_tokens=max_tokens,
            ),
            request_id="prefix-extended",
        )
        extended_prompt_inputs = prepare_prompt_inputs(model_kit, extended_prepared)
        if is_strict_prefix(
            base_prompt_inputs.prompt_input_ids,
            extended_prompt_inputs.prompt_input_ids,
        ):
            return (
                base_prepared,
                extended_prepared,
                base_prompt_inputs,
                extended_prompt_inputs,
                suffix,
            )

    raise RuntimeError(
        "Could not find a strict-prefix prompt extension for prefix mode"
    )


def build_multi_prefix_e2e_requests(
    model_kit,
    processor,
) -> tuple[PreparedRequest, PreparedRequest, object, object, list[int], str]:
    seed = "Reply with exactly one word after reading this filler. "
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa. "

    for repeat_count in range(16, 160):
        prompt_text = build_prompt(processor, seed + unit * repeat_count, None)
        base_prepared = prepare_request_from_prompt_text(
            model_kit,
            prompt_text=prompt_text,
            spec=RequestSpec(
                name="multi-prefix-e2e-base",
                prompt="",
                expected_any=(),
                max_tokens=1,
            ),
            request_id="multi-prefix-e2e-base",
        )
        base_prompt_inputs = prepare_prompt_inputs(model_kit, base_prepared)
        chunks = build_prefix_cache_chunks(
            base_prompt_inputs.prompt_input_ids,
            base_prompt_inputs.image_spans,
        )
        reusable_boundaries = [
            chunk.end
            for chunk in chunks
            if chunk.end < len(base_prompt_inputs.prompt_input_ids)
        ]
        if len(reusable_boundaries) < 2:
            continue
        if len(base_prompt_inputs.prompt_input_ids) > 700:
            break

        for suffix in ["\nA", "\nThe", "\nIt", " .", " ...", "\n"]:
            extended_prepared = prepare_request_from_prompt_text(
                model_kit,
                prompt_text=prompt_text + suffix,
                spec=RequestSpec(
                    name="multi-prefix-e2e-extended",
                    prompt="",
                    expected_any=(),
                    max_tokens=1,
                ),
                request_id="multi-prefix-e2e-extended",
            )
            extended_prompt_inputs = prepare_prompt_inputs(model_kit, extended_prepared)
            if is_strict_prefix(
                base_prompt_inputs.prompt_input_ids,
                extended_prompt_inputs.prompt_input_ids,
            ):
                return (
                    base_prepared,
                    extended_prepared,
                    base_prompt_inputs,
                    extended_prompt_inputs,
                    reusable_boundaries,
                    suffix,
                )

    raise RuntimeError("Could not build a compact two-chunk e2e prompt")


def build_arrays_cache_branch_e2e_requests(
    model_kit,
    processor,
    *,
    min_restored_tokens: int,
) -> tuple[PreparedRequest, PreparedRequest, object, object, int, int, list[int]]:
    user_seed = "Read this previous assistant answer before continuing."
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
    full_user = "Continue from the full answer and reply with one word."
    branch_user = "Now branch from that midpoint and answer with one word."

    for repeat_count in range(64, 240, 8):
        words = (unit * repeat_count).split()
        assistant_text = " ".join(words)
        assistant_prefix = " ".join(words[: len(words) // 2])

        base_prompt_text = processor.apply_chat_template(
            [
                {"role": "user", "content": user_seed},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": full_user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        branch_prompt_text = processor.apply_chat_template(
            [
                {"role": "user", "content": user_seed},
                {"role": "assistant", "content": assistant_prefix},
                {"role": "user", "content": branch_user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        base_prepared = prepare_request_from_prompt_text(
            model_kit,
            prompt_text=base_prompt_text,
            spec=RequestSpec(
                name="arrays-cache-branch-base",
                prompt="",
                expected_any=(),
                max_tokens=1,
            ),
            request_id="arrays-cache-branch-base",
        )
        branch_prepared = prepare_request_from_prompt_text(
            model_kit,
            prompt_text=branch_prompt_text,
            spec=RequestSpec(
                name="arrays-cache-branch-followup",
                prompt="",
                expected_any=(),
                max_tokens=1,
            ),
            request_id="arrays-cache-branch-followup",
        )
        base_prompt_inputs = prepare_prompt_inputs(model_kit, base_prepared)
        branch_prompt_inputs = prepare_prompt_inputs(model_kit, branch_prepared)
        common_prefix_len = common_token_prefix_len(
            base_prompt_inputs.prompt_input_ids,
            branch_prompt_inputs.prompt_input_ids,
        )
        reusable_boundaries = [
            chunk.end
            for chunk in build_prefix_cache_chunks(
                base_prompt_inputs.prompt_input_ids,
                base_prompt_inputs.image_spans,
            )
            if (
                chunk.end <= common_prefix_len
                and chunk.end < len(branch_prompt_inputs.prompt_input_ids)
            )
        ]
        if reusable_boundaries and reusable_boundaries[-1] >= min_restored_tokens:
            return (
                base_prepared,
                branch_prepared,
                base_prompt_inputs,
                branch_prompt_inputs,
                common_prefix_len,
                reusable_boundaries[-1],
                reusable_boundaries,
            )
        if len(base_prompt_inputs.prompt_input_ids) > 1600:
            break

    raise RuntimeError("Could not build a branch prompt with a reusable half-prefix")


def wait_for_prefix_snapshot(
    model_kit,
    *,
    prompt_input_ids: list[int],
    image_spans: list,
    min_chunk_keys: int = 1,
    timeout_s: float,
):
    prompt_cache_store = model_kit._prompt_cache_store

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            image_spans,
        )
        if match is not None and len(restore_chunk_keys(match)) >= min_chunk_keys:
            return match
        time.sleep(0.05)

    raise RuntimeError("Timed out waiting for a reusable prompt snapshot")


def wait_for_restore_plan(
    model_kit,
    *,
    prompt_input_ids: list[int],
    image_spans: list,
    expected_prefix_len: int,
    timeout_s: float,
):
    prompt_cache_store = model_kit._prompt_cache_store

    deadline = time.perf_counter() + timeout_s
    last_seen_prefix = None
    while time.perf_counter() < deadline:
        match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            image_spans,
        )
        if (
            match is not None
            and restore_cached_prefix_len(match) == expected_prefix_len
        ):
            return match
        if match is not None:
            last_seen_prefix = restore_cached_prefix_len(match)
        time.sleep(0.05)

    raise RuntimeError(
        "Timed out waiting for restore plan at "
        f"{expected_prefix_len}; last_seen={last_seen_prefix}"
    )


def find_restore_plan(prompt_cache_store, prompt_input_ids: list[int], image_spans):
    return prompt_cache_store.plan_longest_prefix_restore(
        prompt_input_ids,
        image_spans,
    )


def restore_cached_prefix_len(restore_plan: PromptCacheRestorePlan) -> int:
    return restore_plan.cached_prefix_len


def restore_chunk_keys(restore_plan: PromptCacheRestorePlan) -> list[str]:
    return [chunk.key for chunk in restore_plan.chunks]


def load_restore_plan(prompt_cache_store, restore_plan: PromptCacheRestorePlan):
    return prompt_cache_store.load_restore_plan(restore_plan)


def clear_hot_completed_prompt_cache(model_kit) -> None:
    """Force disk-specific smokes to measure the cache store, not hot MRU."""
    coordinator = getattr(model_kit, "_prompt_cache_coordinator", None)
    if coordinator is None:
        return
    with coordinator._hot_entry_lock:
        coordinator._hot_entry = None


def print_cache_store_stats(prefix: str, prompt_cache_store) -> None:
    stats = prompt_cache_store.snapshot_stats()
    print(f"{prefix}_STATS_TOTAL_BYTES", stats.total_bytes)
    print(f"{prefix}_STATS_MAX_BYTES", stats.max_bytes)
    print(f"{prefix}_STATS_ENTRY_COUNT", stats.entry_count)
    print(f"{prefix}_STATS_HIT_TOKENS", stats.hit_tokens)
    print(f"{prefix}_STATS_MISS_TOKENS", stats.miss_tokens)
    print(f"{prefix}_STATS_EVICTIONS", stats.evictions)


def get_cache_store_record_size(prompt_cache_store, record_key: str) -> int:
    return prompt_cache_store.snapshot_stats().record_sizes_by_key.get(record_key, 0)


def get_cache_store_chunk_size(prompt_cache_store, chunk_key: str) -> int:
    return prompt_cache_store.snapshot_stats().chunk_sizes_by_key.get(chunk_key, 0)


def cache_store_chunk_records_available(prompt_cache_store, chunk_key: str) -> bool:
    return prompt_cache_store.snapshot_stats().chunk_records_available_by_key.get(
        chunk_key,
        False,
    )


def expected_cache_store_record_keys(prompt_cache_store, chunk_key: str) -> list[str]:
    return [
        make_record_key(chunk_key, record_kind)
        for record_kind in RECORD_WRITE_ORDER
        if make_record_key(chunk_key, record_kind)
        in prompt_cache_store._record_metadata_by_key
    ]


def restore_cache_store_record_keys(prompt_cache_store, restore_plan):
    return restore_plan.record_keys_by_chunk_key


def get_stale_optional_record_keys(
    prompt_cache_store,
    restore_plan,
) -> list[str]:
    record_keys_by_chunk = restore_cache_store_record_keys(
        prompt_cache_store,
        restore_plan,
    )
    if record_keys_by_chunk is None:
        return []

    required_record_keys = {
        record_key
        for record_keys in record_keys_by_chunk.values()
        for record_key in record_keys
    }
    stale_record_keys = []
    for chunk_key in restore_chunk_keys(restore_plan):
        for record_key in expected_cache_store_record_keys(
            prompt_cache_store, chunk_key
        ):
            record_metadata = prompt_cache_store._record_metadata_by_key.get(record_key)
            if (
                record_metadata is not None
                and record_metadata.record_kind in OPTIONAL_RECORD_KINDS
                and record_key not in required_record_keys
            ):
                stale_record_keys.append(record_key)

    return stale_record_keys


def wait_for_cache_store_idle(prompt_cache_store, *, timeout_s: float):
    deadline = time.perf_counter() + timeout_s
    last_stats = prompt_cache_store.snapshot_stats()
    stable_under_budget = False
    while time.perf_counter() < deadline:
        last_stats = prompt_cache_store.snapshot_stats()
        under_budget = (
            last_stats.max_bytes is None
            or last_stats.total_bytes <= last_stats.max_bytes
        )
        if under_budget and stable_under_budget:
            return last_stats
        stable_under_budget = under_budget
        time.sleep(0.05)

    raise RuntimeError(f"Cache store did not become idle: {last_stats!r}")


def build_stress_specs(
    *,
    primary_image_b64: str,
    secondary_image_b64: str,
    stagger_s: float,
) -> list[RequestSpec]:
    return [
        RequestSpec(
            name="img-toucan-desc",
            prompt="Describe this image in one short sentence. Include the word bird.",
            expected_any=("bird", "toucan"),
            image_b64=primary_image_b64,
            max_tokens=20,
            top_logprobs=3,
            start_delay_s=0.0,
        ),
        RequestSpec(
            name="txt-alpha",
            prompt="Reply with a short sentence containing the word alpha.",
            expected_any=("alpha",),
            max_tokens=14,
            start_delay_s=0.0,
        ),
        RequestSpec(
            name="img-chameleon-desc",
            prompt="Describe this image in one short sentence. Include the word chameleon.",
            expected_any=("chameleon", "lizard"),
            image_b64=secondary_image_b64,
            max_tokens=20,
            start_delay_s=stagger_s,
        ),
        RequestSpec(
            name="txt-beta",
            prompt="Reply with a short sentence containing the word beta.",
            expected_any=("beta",),
            max_tokens=14,
            top_logprobs=2,
            start_delay_s=stagger_s,
        ),
        RequestSpec(
            name="img-toucan-classify",
            prompt="Classify the animal in one short sentence. Include the word bird.",
            expected_any=("bird",),
            image_b64=primary_image_b64,
            max_tokens=18,
            start_delay_s=stagger_s * 2,
        ),
        RequestSpec(
            name="txt-gamma",
            prompt="Reply with a short sentence containing the word gamma.",
            expected_any=("gamma",),
            max_tokens=14,
            start_delay_s=stagger_s * 2,
        ),
    ]


def run_smoke(model_kit, processor, args, primary_image_b64: str) -> None:
    prepared = prepare_request(
        model_kit,
        processor,
        RequestSpec(
            name="smoke",
            prompt=args.prompt,
            expected_any=("bird", "toucan"),
            image_b64=primary_image_b64,
            max_tokens=args.max_tokens,
        ),
        request_id=args.request_id,
    )
    result = run_prepared_request(model_kit, prepared)
    validate_result(prepared, result)
    print("TEXT", repr(result.text))
    print("STOP_REASON", repr(result.stop_reason))
    print("SMOKE_OK")


def run_stress(
    model_kit,
    processor,
    *,
    primary_image_b64: str,
    secondary_image_b64: str,
    args,
) -> None:
    if args.max_seq_nums < 2:
        raise ValueError("stress mode requires --max-seq-nums >= 2")

    rng = random.Random(args.seed)
    total_requests = 0
    total_top_logprobs_requests = 0

    for round_index in range(args.rounds):
        specs = build_stress_specs(
            primary_image_b64=primary_image_b64,
            secondary_image_b64=secondary_image_b64,
            stagger_s=args.stagger_ms / 1000.0,
        )
        rng.shuffle(specs)

        prepared_requests = [
            prepare_request(
                model_kit,
                processor,
                spec,
                request_id=f"stress-r{round_index + 1}-{i + 1}-{spec.name}",
            )
            for i, spec in enumerate(specs)
        ]

        results: dict[str, RequestResult] = {}
        errors: dict[str, Exception] = {}

        def worker(prepared: PreparedRequest) -> None:
            try:
                results[prepared.request_id] = run_prepared_request(model_kit, prepared)
            except Exception as exc:  # pragma: no cover - exercised in live stress mode
                errors[prepared.request_id] = exc

        threads = [
            threading.Thread(target=worker, args=(prepared,), daemon=True)
            for prepared in prepared_requests
        ]

        start_time = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.join_timeout_s)
        wall_time = time.perf_counter() - start_time

        alive_request_ids = [
            prepared.request_id
            for prepared, thread in zip(prepared_requests, threads)
            if thread.is_alive()
        ]
        if alive_request_ids:
            raise RuntimeError(
                f"Round {round_index + 1} timed out waiting for {alive_request_ids}"
            )
        if errors:
            raise RuntimeError(
                f"Round {round_index + 1} encountered errors: {errors!r}"
            )
        if len(results) != len(prepared_requests):
            raise RuntimeError(
                f"Round {round_index + 1} returned {len(results)} results for {len(prepared_requests)} requests"
            )

        print(f"ROUND {round_index + 1} wall_time={wall_time:.2f}s")
        for prepared in prepared_requests:
            result = results[prepared.request_id]
            validate_result(prepared, result)
            total_requests += 1
            if prepared.top_logprobs > 0:
                total_top_logprobs_requests += 1
            print(
                "  "
                f"{prepared.request_id} "
                f"elapsed={result.elapsed_s:.2f}s "
                f"stop={result.stop_reason} "
                f"text={truncate_text(result.text)!r}"
            )

    print("ROUNDS", args.rounds)
    print("REQUESTS", total_requests)
    print("TOP_LOGPROBS_REQUESTS", total_top_logprobs_requests)
    print("STRESS_OK")


def run_prefix(
    model_kit,
    processor,
    *,
    primary_image_b64: str,
    secondary_image_b64: str,
    args,
    expect_warm_base: bool = False,
) -> None:
    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        suffix,
    ) = build_prefix_requests(
        model_kit,
        processor,
        prompt=args.prompt,
        primary_image_b64=primary_image_b64,
    )
    control_prepared = prepare_request_from_prompt_text(
        model_kit,
        prompt_text=build_prompt(processor, args.prompt, secondary_image_b64),
        spec=RequestSpec(
            name="prefix-control",
            prompt=args.prompt,
            expected_any=("chameleon", "lizard"),
            image_b64=secondary_image_b64,
            max_tokens=24,
        ),
        request_id="prefix-control",
    )
    control_prompt_inputs = prepare_prompt_inputs(model_kit, control_prepared)

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    validate_result(base_prepared, base_result)

    prompt_cache_store = model_kit._prompt_cache_store
    disk_cache_store_enabled = prompt_cache_store.can_store_records()
    if disk_cache_store_enabled:
        prefix_match = wait_for_prefix_snapshot(
            model_kit,
            prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
            image_spans=extended_prompt_inputs.image_spans,
            timeout_s=args.prefix_wait_timeout_s,
        )
        expected_prefix_len = restore_cached_prefix_len(prefix_match)
        control_prefix_match = find_restore_plan(
            prompt_cache_store,
            control_prompt_inputs.prompt_input_ids,
            control_prompt_inputs.image_spans,
        )
    else:
        expected_prefix_len = 0
        control_prefix_match = None
    expected_control_progress = (
        0
        if control_prefix_match is None
        else restore_cached_prefix_len(control_prefix_match)
    )

    extended_trace = PromptProgressTrace()
    extended_result = run_prepared_request(
        model_kit,
        extended_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(extended_trace),
    )
    if extended_result.stop_reason is None:
        raise RuntimeError("Extended request completed without a stop reason")

    control_trace = PromptProgressTrace()
    control_result = run_prepared_request(
        model_kit,
        control_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(control_trace),
    )
    validate_result(control_prepared, control_result)

    if base_trace.begin_prefill_tokens_processed is None:
        raise RuntimeError("Base request did not emit a prompt-progress begin event")
    if extended_trace.begin_prefill_tokens_processed is None:
        raise RuntimeError(
            "Extended request did not emit a prompt-progress begin event"
        )
    if control_trace.begin_prefill_tokens_processed is None:
        raise RuntimeError("Control request did not emit a prompt-progress begin event")
    expected_base_progress = (
        len(base_prompt_inputs.prompt_input_ids) - 1 if expect_warm_base else 0
    )
    if base_trace.begin_prefill_tokens_processed != expected_base_progress:
        expectation = "warm" if expect_warm_base else "cold"
        raise RuntimeError(
            "Base request did not match expected restart state: "
            f"{base_trace.begin_prefill_tokens_processed} != {expected_base_progress} "
            f"({expectation})"
        )
    expected_extended_progress = (
        len(extended_prompt_inputs.prompt_input_ids) - 1
        if expect_warm_base
        else expected_prefix_len
    )
    hot_completed_progress = min(
        len(base_prompt_inputs.prompt_input_ids),
        len(extended_prompt_inputs.prompt_input_ids) - 1,
    )
    accepted_extended_progress = {expected_extended_progress}
    if not expect_warm_base:
        accepted_extended_progress.add(hot_completed_progress)
    if extended_trace.begin_prefill_tokens_processed not in accepted_extended_progress:
        raise RuntimeError(
            "Extended request did not restore the expected cached prefix: "
            f"{extended_trace.begin_prefill_tokens_processed} not in "
            f"{sorted(accepted_extended_progress)}"
        )
    hot_control_progress = image_safe_common_prefix_len(
        control_prompt_inputs.prompt_input_ids,
        control_prompt_inputs.image_spans,
        extended_prompt_inputs.prompt_input_ids,
        extended_prompt_inputs.image_spans,
        max_prefix_len=len(control_prompt_inputs.prompt_input_ids) - 1,
    )
    accepted_control_progress = {expected_control_progress, hot_control_progress}
    if control_trace.begin_prefill_tokens_processed not in accepted_control_progress:
        raise RuntimeError(
            "Control request did not match expected restart state: "
            f"{control_trace.begin_prefill_tokens_processed} not in "
            f"{sorted(accepted_control_progress)}"
        )

    print("PREFIX_SUFFIX", repr(suffix))
    print("PREFIX_BASE_EXPECTED_WARM", expect_warm_base)
    print("PREFIX_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids))
    print("PREFIX_EXTENDED_PROMPT_TOKENS", len(extended_prompt_inputs.prompt_input_ids))
    print("PREFIX_RESTORED_TOKENS", extended_trace.begin_prefill_tokens_processed)
    print("PREFIX_DISK_RESTORED_TOKENS", expected_prefix_len)
    print("PREFIX_HOT_RESTORED_TOKENS", hot_completed_progress)
    print("PREFIX_CONTROL_RESTORED_TOKENS", expected_control_progress)
    print("PREFIX_CONTROL_HOT_RESTORED_TOKENS", hot_control_progress)
    print(
        "BASE_BEGIN",
        base_trace.begin_cached_tokens,
        base_trace.begin_total_prompt_tokens,
        base_trace.begin_prefill_tokens_processed,
    )
    print(
        "EXTENDED_BEGIN",
        extended_trace.begin_cached_tokens,
        extended_trace.begin_total_prompt_tokens,
        extended_trace.begin_prefill_tokens_processed,
    )
    print(
        "CONTROL_BEGIN",
        control_trace.begin_cached_tokens,
        control_trace.begin_total_prompt_tokens,
        control_trace.begin_prefill_tokens_processed,
    )
    print("BASE_TEXT", repr(truncate_text(base_result.text)))
    print("EXTENDED_TEXT", repr(truncate_text(extended_result.text)))
    print("CONTROL_TEXT", repr(truncate_text(control_result.text)))
    print("PREFIX_RESTART_OK" if expect_warm_base else "PREFIX_OK")


def make_synthetic_prompt_cache(prefix_len: int):
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    values = keys + 1000
    kv_cache = KVCache()
    kv_cache.state = (keys, values)
    arrays_cache = ArraysCache(size=1)
    arrays_cache[0] = mx.array([[prefix_len]], dtype=mx.int32)
    return [kv_cache, arrays_cache]


def make_synthetic_kv_prompt_cache(prefix_len: int):
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    values = keys + 1000
    kv_cache = KVCache()
    kv_cache.state = (keys, values)
    return [kv_cache]


def make_synthetic_rotating_prompt_cache(prefix_len: int):
    keys = mx.arange(prefix_len, dtype=mx.float32).reshape(1, 1, prefix_len, 1)
    values = keys + 1000
    kv_cache = KVCache()
    kv_cache.state = (keys, values)

    window_size = 512
    window_start = max(0, prefix_len - window_size)
    rotating_keys = mx.arange(window_start, prefix_len, dtype=mx.float32).reshape(
        1, 1, prefix_len - window_start, 1
    )
    rotating_cache = RotatingKVCache(max_size=window_size, keep=0)
    rotating_cache.state = (rotating_keys, rotating_keys + 2000)
    rotating_cache.offset = prefix_len
    rotating_cache._idx = rotating_keys.shape[2]
    return [kv_cache, rotating_cache]


def save_synthetic_chunk(
    prompt_cache_store: VlmPromptCacheStore,
    chunk,
    prompt_input_ids: list[int],
    *,
    prompt_cache=None,
) -> None:
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    chunk_idx = next(idx for idx, item in enumerate(chunks) if item.key == chunk.key)
    pending_save = prompt_cache_store.prepare_save(
        chunk=chunk,
        prefix_chunks=chunks[: chunk_idx + 1],
        prompt_cache=prompt_cache or make_synthetic_prompt_cache(chunk.end),
    )
    if pending_save is None:
        raise RuntimeError(f"Could not prepare chunk save at {chunk.end}")
    prompt_cache_store.commit_pending_save(pending_save)


def run_multi_prefix(args) -> None:
    base_image_prompt = list(range(20)) + [999] * 3 + list(range(20, 300))
    extended_image_prompt = base_image_prompt + [999] * 2 + list(range(300, 620))
    base_image_chunks = build_prefix_cache_chunks(
        base_image_prompt,
        [PromptImageSpan(20, 23, "image-a")],
    )
    extended_image_chunks = build_prefix_cache_chunks(
        extended_image_prompt,
        [
            PromptImageSpan(20, 23, "image-a"),
            PromptImageSpan(
                len(base_image_prompt),
                len(base_image_prompt) + 2,
                "image-b",
            ),
        ],
    )
    changed_image_chunks = build_prefix_cache_chunks(
        base_image_prompt,
        [PromptImageSpan(20, 23, "image-c")],
    )
    base_image_keys = [chunk.key for chunk in base_image_chunks]
    extended_shared_keys = [
        chunk.key
        for chunk in extended_image_chunks
        if chunk.end <= base_image_chunks[-1].end
    ]
    if extended_shared_keys != base_image_keys:
        raise RuntimeError("Appending an image changed earlier prompt chunk keys")
    if changed_image_chunks[0].key == base_image_chunks[0].key:
        raise RuntimeError("Changing an image did not change its containing chunk")
    image_reuse_prefix_len = image_safe_common_prefix_len(
        extended_image_prompt,
        [
            PromptImageSpan(20, 23, "image-a"),
            PromptImageSpan(
                len(base_image_prompt),
                len(base_image_prompt) + 2,
                "image-b",
            ),
        ],
        base_image_prompt,
        [PromptImageSpan(20, 23, "image-a")],
        max_prefix_len=len(base_image_prompt),
    )
    if image_reuse_prefix_len != len(base_image_prompt):
        raise RuntimeError(
            "Hot image-prefix reuse did not preserve the matching image prefix: "
            f"{image_reuse_prefix_len} != {len(base_image_prompt)}"
        )

    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    reusable_chunks = [chunk for chunk in chunks if chunk.end < len(prompt_input_ids)]
    if len(reusable_chunks) < 2:
        raise RuntimeError("Synthetic prompt did not produce two reusable chunks")

    prompt_cache_store = VlmPromptCacheStore()
    try:
        for chunk in reusable_chunks[:2]:
            save_synthetic_chunk(prompt_cache_store, chunk, prompt_input_ids)

        prefix_match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if prefix_match is None or len(restore_chunk_keys(prefix_match)) != 2:
            raise RuntimeError("Synthetic lookup did not return two chunk keys")
        expected_prefix_len = reusable_chunks[1].end
        if restore_cached_prefix_len(prefix_match) != expected_prefix_len:
            raise RuntimeError(
                "Synthetic lookup matched the wrong boundary: "
                f"{restore_cached_prefix_len(prefix_match)} != {expected_prefix_len}"
            )

        record_keys_by_chunk = restore_cache_store_record_keys(
            prompt_cache_store,
            prefix_match,
        )
        if record_keys_by_chunk is None:
            raise RuntimeError("Synthetic record selection failed")
        state_record_keys = [
            record_key
            for record_keys in record_keys_by_chunk.values()
            for record_key in record_keys
            if (
                prompt_cache_store._record_metadata_by_key[record_key].record_kind
                == RECORD_KIND_STATE_CHECKPOINT
            )
        ]
        if len(state_record_keys) != 1:
            raise RuntimeError(
                f"Expected one final state checkpoint record, got {len(state_record_keys)}"
            )

        first_chunk_state_record_key = next(
            record_key
            for record_key in expected_cache_store_record_keys(
                prompt_cache_store,
                restore_chunk_keys(prefix_match)[0],
            )
            if (
                prompt_cache_store._record_metadata_by_key[record_key].record_kind
                == RECORD_KIND_STATE_CHECKPOINT
            )
        )
        if first_chunk_state_record_key in state_record_keys:
            raise RuntimeError("Record selection kept an old state checkpoint")

        first_state_size = get_cache_store_record_size(
            prompt_cache_store,
            first_chunk_state_record_key,
        )
        total_before_eviction = prompt_cache_store._total_bytes
        prompt_cache_store._max_cache_store_bytes = (
            total_before_eviction - first_state_size + 1
        )
        prompt_cache_store._evict_if_needed()
        if first_chunk_state_record_key in prompt_cache_store._record_metadata_by_key:
            raise RuntimeError("Eviction did not choose the old state checkpoint first")
        if prompt_cache_store._total_bytes > prompt_cache_store._max_cache_store_bytes:
            raise RuntimeError(
                "State checkpoint eviction did not enforce the byte budget"
            )

        prefix_match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if prefix_match is None or len(restore_chunk_keys(prefix_match)) != 2:
            raise RuntimeError(
                "Synthetic lookup required an evicted old state checkpoint"
            )

        stored_state = load_restore_plan(prompt_cache_store, prefix_match)

        kv_keys, kv_values = stored_state.prompt_cache[0].state
        arrays_value = stored_state.prompt_cache[1][0]
        mx.eval(kv_keys, kv_values, arrays_value)
        if kv_keys.shape[2] != expected_prefix_len:
            raise RuntimeError(
                f"Assembled KV length {kv_keys.shape[2]} != {expected_prefix_len}"
            )
        if kv_keys[0, 0, 0, 0].item() != 0 or kv_keys[0, 0, -1, 0].item() != 511:
            raise RuntimeError("Assembled KV keys did not preserve chunk order")
        if kv_values[0, 0, -1, 0].item() != 1511:
            raise RuntimeError("Assembled KV values did not preserve chunk order")
        if arrays_value[0, 0].item() != expected_prefix_len:
            raise RuntimeError("Boundary cache layer did not use the latest chunk")

        cache_file_sizes = prompt_cache_store.snapshot_stats().record_sizes
        print("MULTI_PREFIX_SYNTHETIC", True)
        print("MULTI_PREFIX_CHUNK_KEYS", len(restore_chunk_keys(prefix_match)))
        print("MULTI_PREFIX_BOUNDARIES", [chunk.end for chunk in reusable_chunks[:2]])
        print("MULTI_PREFIX_RESTORED_TOKENS", expected_prefix_len)
        print("MULTI_PREFIX_EVICTED_OLD_STATE", True)
        print("MULTI_PREFIX_TOTAL_BEFORE_EVICT", total_before_eviction)
        print("MULTI_PREFIX_TOTAL_AFTER_EVICT", prompt_cache_store._total_bytes)
        print("MULTI_PREFIX_CACHE_FILES", len(cache_file_sizes))
        print("MULTI_PREFIX_CACHE_FILE_SIZES", cache_file_sizes)
        print("MULTI_PREFIX_OK")
    finally:
        prompt_cache_store.close()


def run_rotating_suffix(args) -> None:
    prompt_input_ids = list(range(900))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    reusable_chunks = [chunk for chunk in chunks if chunk.end < len(prompt_input_ids)]
    if len(reusable_chunks) < 3:
        raise RuntimeError("Synthetic prompt did not produce three reusable chunks")

    prompt_cache_store = VlmPromptCacheStore()
    try:
        for chunk in reusable_chunks[:3]:
            save_synthetic_chunk(
                prompt_cache_store,
                chunk,
                prompt_input_ids,
                prompt_cache=make_synthetic_rotating_prompt_cache(chunk.end),
            )

        prefix_match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if prefix_match is None or len(restore_chunk_keys(prefix_match)) != 3:
            raise RuntimeError("Rotating lookup did not return three chunk keys")

        record_keys_by_chunk = restore_cache_store_record_keys(
            prompt_cache_store,
            prefix_match,
        )
        if record_keys_by_chunk is None:
            raise RuntimeError("Rotating record selection failed")
        rotating_record_keys = [
            record_key
            for record_keys in record_keys_by_chunk.values()
            for record_key in record_keys
            if (
                prompt_cache_store._record_metadata_by_key[record_key].record_kind
                == RECORD_KIND_ROTATING_DELTA
            )
        ]
        if len(rotating_record_keys) != 2:
            raise RuntimeError(
                f"Expected two rotating suffix records, got {len(rotating_record_keys)}"
            )

        first_chunk_rotating_record_key = next(
            record_key
            for record_key in expected_cache_store_record_keys(
                prompt_cache_store,
                restore_chunk_keys(prefix_match)[0],
            )
            if (
                prompt_cache_store._record_metadata_by_key[record_key].record_kind
                == RECORD_KIND_ROTATING_DELTA
            )
        )
        first_rotating_size = get_cache_store_record_size(
            prompt_cache_store,
            first_chunk_rotating_record_key,
        )
        total_before_eviction = prompt_cache_store._total_bytes
        prompt_cache_store._max_cache_store_bytes = (
            total_before_eviction - first_rotating_size + 1
        )
        prompt_cache_store._evict_if_needed()
        if (
            first_chunk_rotating_record_key
            in prompt_cache_store._record_metadata_by_key
        ):
            raise RuntimeError("Eviction did not choose the old SWA record first")
        if prompt_cache_store._total_bytes > prompt_cache_store._max_cache_store_bytes:
            raise RuntimeError("Rotating eviction did not enforce the byte budget")

        prefix_match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if prefix_match is None or len(restore_chunk_keys(prefix_match)) != 3:
            raise RuntimeError(
                "Rotating lookup required an evicted record outside the final window"
            )
        record_keys_by_chunk = restore_cache_store_record_keys(
            prompt_cache_store,
            prefix_match,
        )
        if record_keys_by_chunk is None:
            raise RuntimeError("Rotating record selection failed after eviction")
        if first_chunk_rotating_record_key in [
            record_key
            for record_keys in record_keys_by_chunk.values()
            for record_key in record_keys
        ]:
            raise RuntimeError("Rotating record selection kept an old SWA record")

        stored_state = load_restore_plan(prompt_cache_store, prefix_match)
        kv_keys, _ = stored_state.prompt_cache[0].state
        rotating_keys, _ = stored_state.prompt_cache[1].state
        mx.eval(kv_keys, rotating_keys)
        if kv_keys.shape[2] != reusable_chunks[2].end:
            raise RuntimeError("Rotating suffix restore assembled the wrong KV length")
        if rotating_keys.shape[2] != 512:
            raise RuntimeError("Rotating suffix restore assembled the wrong SWA length")
        if (
            rotating_keys[0, 0, 0, 0].item() != 256
            or rotating_keys[0, 0, -1, 0].item() != 767
        ):
            raise RuntimeError("Rotating suffix restore did not use the last window")

        print("ROTATING_SUFFIX_CHUNK_KEYS", len(restore_chunk_keys(prefix_match)))
        print("ROTATING_SUFFIX_RECORDS", len(rotating_record_keys))
        print("ROTATING_SUFFIX_EVICTED_OLD_RECORD", True)
        print("ROTATING_SUFFIX_TOTAL_BEFORE_EVICT", total_before_eviction)
        print("ROTATING_SUFFIX_TOTAL_AFTER_EVICT", prompt_cache_store._total_bytes)
        print("ROTATING_SUFFIX_KV_TOKENS", kv_keys.shape[2])
        print("ROTATING_SUFFIX_WINDOW_TOKENS", rotating_keys.shape[2])
        print("ROTATING_SUFFIX_OK")
    finally:
        prompt_cache_store.close()


def run_eviction(args) -> None:
    prompt_input_ids = list(range(600))
    chunks = build_prefix_cache_chunks(prompt_input_ids, [])
    reusable_chunks = [chunk for chunk in chunks if chunk.end < len(prompt_input_ids)]
    if len(reusable_chunks) < 2:
        raise RuntimeError("Synthetic prompt did not produce two reusable chunks")

    prompt_cache_store = VlmPromptCacheStore()
    try:
        first_chunk, second_chunk = reusable_chunks[:2]
        save_synthetic_chunk(
            prompt_cache_store,
            first_chunk,
            prompt_input_ids,
            prompt_cache=make_synthetic_kv_prompt_cache(first_chunk.end),
        )
        first_size = get_cache_store_chunk_size(prompt_cache_store, first_chunk.key)
        prompt_cache_store._max_cache_store_bytes = (
            args.cache_max_bytes or first_size + 64
        )

        save_synthetic_chunk(
            prompt_cache_store,
            second_chunk,
            prompt_input_ids,
            prompt_cache=make_synthetic_kv_prompt_cache(second_chunk.end),
        )
        first_exists = cache_store_chunk_records_available(
            prompt_cache_store,
            first_chunk.key,
        )
        second_exists = cache_store_chunk_records_available(
            prompt_cache_store,
            second_chunk.key,
        )
        if not first_exists:
            raise RuntimeError("Eviction removed the prefix chunk before its dependent")
        if second_exists:
            raise RuntimeError("Eviction did not remove the dependent leaf chunk")
        if prompt_cache_store._total_bytes > prompt_cache_store._max_cache_store_bytes:
            raise RuntimeError("Eviction did not enforce the byte budget")

        best_effort_restore = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if best_effort_restore is None:
            raise RuntimeError("Eviction best-effort restore missed")
        best_effort_state = load_restore_plan(
            prompt_cache_store,
            best_effort_restore,
        )
        if best_effort_state.cached_prefix_len != first_chunk.end:
            raise RuntimeError("Eviction best-effort restore used the wrong prefix")

        prefix_match = find_restore_plan(
            prompt_cache_store,
            prompt_input_ids,
            [],
        )
        if (
            prefix_match is None
            or restore_cached_prefix_len(prefix_match) != first_chunk.end
        ):
            raise RuntimeError("Eviction did not preserve the shorter reusable prefix")

        kv_keys, _ = best_effort_state.prompt_cache[0].state
        mx.eval(kv_keys)
        if kv_keys.shape[2] != first_chunk.end:
            raise RuntimeError("Eviction restored the wrong prefix length")

        cache_files = prompt_cache_store.snapshot_stats().record_sizes
        print("EVICTION_MAX_BYTES", prompt_cache_store._max_cache_store_bytes)
        print("EVICTION_TOTAL_BYTES", prompt_cache_store._total_bytes)
        print("EVICTION_FILES", len(cache_files))
        print("EVICTION_FIRST_EXISTS", first_exists)
        print("EVICTION_SECOND_EXISTS", second_exists)
        print("EVICTION_RESTORED_TOKENS", restore_cached_prefix_len(prefix_match))
        print("EVICTION_BEST_EFFORT_TOKENS", best_effort_state.cached_prefix_len)
        print("EVICTION_OK")
    finally:
        prompt_cache_store.close()


def run_multi_prefix_e2e(model_kit, processor, args) -> None:
    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        reusable_boundaries,
        suffix,
    ) = build_multi_prefix_e2e_requests(model_kit, processor)
    prompt_cache_store = model_kit._prompt_cache_store

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    if base_result.stop_reason is None:
        raise RuntimeError("Base e2e request completed without a stop reason")

    prefix_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
        image_spans=extended_prompt_inputs.image_spans,
        min_chunk_keys=2,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_restored_tokens = reusable_boundaries[1]
    if restore_cached_prefix_len(prefix_match) != expected_restored_tokens:
        raise RuntimeError(
            "E2E prefix lookup did not stop at the second chunk boundary: "
            f"{restore_cached_prefix_len(prefix_match)} != {expected_restored_tokens}"
        )

    clear_hot_completed_prompt_cache(model_kit)
    extended_trace = PromptProgressTrace()
    extended_result = run_prepared_request(
        model_kit,
        extended_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(extended_trace),
    )
    if extended_result.stop_reason is None:
        raise RuntimeError("Extended e2e request completed without a stop reason")
    if extended_trace.begin_prefill_tokens_processed != expected_restored_tokens:
        raise RuntimeError(
            "E2E request did not restore the expected cached prefix: "
            f"{extended_trace.begin_prefill_tokens_processed} != {expected_restored_tokens}"
        )

    print("MULTI_PREFIX_E2E_SUFFIX", repr(suffix))
    print(
        "MULTI_PREFIX_E2E_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids)
    )
    print(
        "MULTI_PREFIX_E2E_EXTENDED_PROMPT_TOKENS",
        len(extended_prompt_inputs.prompt_input_ids),
    )
    print("MULTI_PREFIX_E2E_BOUNDARIES", reusable_boundaries)
    print("MULTI_PREFIX_E2E_CHUNK_KEYS", len(restore_chunk_keys(prefix_match)))
    print("MULTI_PREFIX_E2E_RESTORED_TOKENS", expected_restored_tokens)
    print(
        "MULTI_PREFIX_E2E_BASE_BEGIN",
        base_trace.begin_cached_tokens,
        base_trace.begin_total_prompt_tokens,
        base_trace.begin_prefill_tokens_processed,
    )
    print(
        "MULTI_PREFIX_E2E_EXTENDED_BEGIN",
        extended_trace.begin_cached_tokens,
        extended_trace.begin_total_prompt_tokens,
        extended_trace.begin_prefill_tokens_processed,
    )
    print("MULTI_PREFIX_E2E_BASE_TEXT", repr(truncate_text(base_result.text)))
    print("MULTI_PREFIX_E2E_EXTENDED_TEXT", repr(truncate_text(extended_result.text)))
    print_cache_store_stats("MULTI_PREFIX_E2E", prompt_cache_store)
    print("MULTI_PREFIX_E2E_OK")


def run_hybrid_cache_e2e(
    model_kit,
    processor,
    args,
    *,
    required_cache_class: str | None = None,
) -> None:
    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        reusable_boundaries,
        suffix,
    ) = build_multi_prefix_e2e_requests(model_kit, processor)
    prompt_cache_store = model_kit._prompt_cache_store

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    if base_result.stop_reason is None:
        raise RuntimeError("Base hybrid-cache request completed without a stop reason")

    prefix_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
        image_spans=extended_prompt_inputs.image_spans,
        min_chunk_keys=2,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_restored_tokens = reusable_boundaries[1]
    if restore_cached_prefix_len(prefix_match) != expected_restored_tokens:
        raise RuntimeError(
            "Hybrid cache lookup did not stop at the second chunk boundary: "
            f"{restore_cached_prefix_len(prefix_match)} != {expected_restored_tokens}"
        )

    per_chunk_record_counts = []
    total_record_counts = Counter()
    for chunk_key in restore_chunk_keys(prefix_match):
        chunk_record_counts = Counter()
        for record_key in expected_cache_store_record_keys(
            prompt_cache_store, chunk_key
        ):
            metadata = prompt_cache_store._record_metadata_by_key[record_key]
            chunk_record_counts[metadata.record_kind] += len(metadata.layer_indices)
        per_chunk_record_counts.append(dict(chunk_record_counts))
        total_record_counts.update(chunk_record_counts)
    if total_record_counts[RECORD_KIND_KV_DELTA] == 0:
        raise RuntimeError("Hybrid cache did not save any KV delta records")
    if (
        total_record_counts[RECORD_KIND_STATE_CHECKPOINT] == 0
        and total_record_counts[RECORD_KIND_ROTATING_DELTA] == 0
    ):
        raise RuntimeError("Hybrid cache did not save any non-full-attention records")

    stored_state = load_restore_plan(prompt_cache_store, prefix_match)
    restored_cache_classes = [
        type(cache).__name__ for cache in stored_state.prompt_cache
    ]
    restored_cache_counts = dict(Counter(restored_cache_classes))
    if "KVCache" not in restored_cache_counts:
        raise RuntimeError("Hybrid restore did not include KVCache layers")
    if (
        "ArraysCache" not in restored_cache_counts
        and "RotatingKVCache" not in restored_cache_counts
    ):
        raise RuntimeError(
            "Hybrid restore did not include ArraysCache or RotatingKVCache layers"
        )
    if (
        required_cache_class is not None
        and required_cache_class not in restored_cache_counts
    ):
        raise RuntimeError(
            f"Hybrid restore did not include required {required_cache_class} layers"
        )

    clear_hot_completed_prompt_cache(model_kit)
    extended_trace = PromptProgressTrace()
    extended_result = run_prepared_request(
        model_kit,
        extended_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(extended_trace),
    )
    if extended_result.stop_reason is None:
        raise RuntimeError(
            "Extended hybrid-cache request completed without a stop reason"
        )
    if extended_trace.begin_prefill_tokens_processed != expected_restored_tokens:
        raise RuntimeError(
            "Hybrid e2e request did not restore the expected cached prefix: "
            f"{extended_trace.begin_prefill_tokens_processed} != {expected_restored_tokens}"
        )

    print("HYBRID_CACHE_SUFFIX", repr(suffix))
    print("HYBRID_CACHE_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids))
    print(
        "HYBRID_CACHE_EXTENDED_PROMPT_TOKENS",
        len(extended_prompt_inputs.prompt_input_ids),
    )
    print("HYBRID_CACHE_BOUNDARIES", reusable_boundaries)
    print("HYBRID_CACHE_CHUNK_KEYS", len(restore_chunk_keys(prefix_match)))
    print("HYBRID_CACHE_RESTORED_TOKENS", expected_restored_tokens)
    print("HYBRID_CACHE_RECORD_COUNTS", dict(total_record_counts))
    print("HYBRID_CACHE_PER_CHUNK_RECORD_COUNTS", per_chunk_record_counts)
    print("HYBRID_CACHE_RESTORED_CACHE_COUNTS", restored_cache_counts)
    print(
        "HYBRID_CACHE_BASE_BEGIN",
        base_trace.begin_cached_tokens,
        base_trace.begin_total_prompt_tokens,
        base_trace.begin_prefill_tokens_processed,
    )
    print(
        "HYBRID_CACHE_EXTENDED_BEGIN",
        extended_trace.begin_cached_tokens,
        extended_trace.begin_total_prompt_tokens,
        extended_trace.begin_prefill_tokens_processed,
    )
    print("HYBRID_CACHE_BASE_TEXT", repr(truncate_text(base_result.text)))
    print("HYBRID_CACHE_EXTENDED_TEXT", repr(truncate_text(extended_result.text)))
    print_cache_store_stats("HYBRID_CACHE", prompt_cache_store)
    print("HYBRID_CACHE_E2E_OK")


def run_arrays_cache_branch_e2e(model_kit, processor, args) -> None:
    (
        base_prepared,
        branch_prepared,
        base_prompt_inputs,
        branch_prompt_inputs,
        common_prefix_len,
        expected_restored_tokens,
        reusable_boundaries,
    ) = build_arrays_cache_branch_e2e_requests(
        model_kit,
        processor,
        min_restored_tokens=512 if args.cache_max_bytes is not None else 256,
    )
    prompt_cache_store = model_kit._prompt_cache_store
    if args.cache_max_bytes is not None:
        prompt_cache_store._max_cache_store_bytes = args.cache_max_bytes
        prompt_cache_store._empirical_budget_set = True

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    if base_result.stop_reason is None:
        raise RuntimeError("Branch base request completed without a stop reason")

    prefix_match = wait_for_restore_plan(
        model_kit,
        prompt_input_ids=branch_prompt_inputs.prompt_input_ids,
        image_spans=branch_prompt_inputs.image_spans,
        expected_prefix_len=expected_restored_tokens,
        timeout_s=args.prefix_wait_timeout_s,
    )
    planned_record_kind_counts = Counter(
        prompt_cache_store._record_metadata_by_key[record_key].record_kind
        for record_keys in prefix_match[2].values()
        for record_key in record_keys
    )
    if planned_record_kind_counts[RECORD_KIND_STATE_CHECKPOINT] == 0:
        raise RuntimeError("Branch restore plan did not require ArraysCache checkpoint")

    clear_hot_completed_prompt_cache(model_kit)
    branch_trace = PromptProgressTrace()
    branch_result = run_prepared_request(
        model_kit,
        branch_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(branch_trace),
    )
    if branch_result.stop_reason is None:
        raise RuntimeError("Branch follow-up completed without a stop reason")
    if branch_trace.begin_prefill_tokens_processed != expected_restored_tokens:
        raise RuntimeError(
            "Branch follow-up did not resume from the nearest cached boundary: "
            f"{branch_trace.begin_prefill_tokens_processed} "
            f"!= {expected_restored_tokens}"
        )

    print("ARRAYS_BRANCH_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids))
    print(
        "ARRAYS_BRANCH_FOLLOWUP_PROMPT_TOKENS",
        len(branch_prompt_inputs.prompt_input_ids),
    )
    print("ARRAYS_BRANCH_COMMON_PREFIX_TOKENS", common_prefix_len)
    print("ARRAYS_BRANCH_BOUNDARIES", reusable_boundaries)
    print("ARRAYS_BRANCH_RESTORED_TOKENS", expected_restored_tokens)
    print("ARRAYS_BRANCH_RESTORE_RECORD_KIND_COUNTS", dict(planned_record_kind_counts))
    print(
        "ARRAYS_BRANCH_BASE_BEGIN",
        base_trace.begin_cached_tokens,
        base_trace.begin_total_prompt_tokens,
        base_trace.begin_prefill_tokens_processed,
    )
    print(
        "ARRAYS_BRANCH_FOLLOWUP_BEGIN",
        branch_trace.begin_cached_tokens,
        branch_trace.begin_total_prompt_tokens,
        branch_trace.begin_prefill_tokens_processed,
    )
    print("ARRAYS_BRANCH_BASE_TEXT", repr(truncate_text(base_result.text)))
    print("ARRAYS_BRANCH_FOLLOWUP_TEXT", repr(truncate_text(branch_result.text)))
    print_cache_store_stats("ARRAYS_BRANCH", prompt_cache_store)
    print("ARRAYS_CACHE_BRANCH_E2E_OK")


def _qwen_language_model(model_kit):
    language_model = getattr(model_kit.model, "language_model", None)
    if language_model is None or not hasattr(language_model, "_rope_deltas"):
        raise RuntimeError("Loaded model does not expose Qwen-style rope state")
    return language_model


def _clear_qwen_rope_state(model_kit) -> None:
    language_model = _qwen_language_model(model_kit)
    language_model._position_ids = None
    language_model._rope_deltas = None


def run_qwen_rope_e2e(
    model_kit,
    processor,
    *,
    primary_image_b64: str,
    args,
) -> None:
    filler = "alpha beta gamma delta epsilon zeta eta theta. " * 32
    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        _suffix,
    ) = build_prefix_requests(
        model_kit,
        processor,
        prompt=f"{args.prompt} {filler}",
        primary_image_b64=primary_image_b64,
        max_tokens=min(args.max_tokens, 4),
    )
    if not [
        chunk
        for chunk in build_prefix_cache_chunks(
            base_prompt_inputs.prompt_input_ids,
            base_prompt_inputs.image_spans,
        )
        if chunk.end < len(base_prompt_inputs.prompt_input_ids)
    ]:
        raise RuntimeError("Qwen rope prompt did not produce a restorable chunk")

    base_result = run_prepared_request(model_kit, base_prepared)
    if base_result.stop_reason is None:
        raise RuntimeError("Qwen rope base request completed without a stop reason")

    prefix_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
        image_spans=extended_prompt_inputs.image_spans,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_restored_tokens = restore_cached_prefix_len(prefix_match)

    clear_hot_completed_prompt_cache(model_kit)
    _clear_qwen_rope_state(model_kit)

    coordinator = getattr(model_kit, "_prompt_cache_coordinator", None)
    if coordinator is None:
        raise RuntimeError("qwen-rope-e2e mode requires the cache coordinator")
    language_model = _qwen_language_model(model_kit)
    original_restore = coordinator.restore
    original_get_input_embeddings = model_kit.model.get_input_embeddings
    restore_probe = {}
    embedding_probe = {}

    def restore_wrapper(*, prompt_input_ids: list[int], image_spans: list):
        restored = original_restore(
            prompt_input_ids=prompt_input_ids,
            image_spans=image_spans,
        )
        if prompt_input_ids == extended_prompt_inputs.prompt_input_ids:
            restore_probe["called"] = True
            restore_probe["cached_prefix_len"] = (
                None if restored is None else restored.cached_prefix_len
            )
            restore_probe["rope_deltas_is_none"] = (
                restored is not None and restored.rope_deltas is None
            )
        return restored

    def get_input_embeddings_wrapper(*args_, **kwargs):
        input_ids = args_[0] if args_ else kwargs.get("input_ids")
        pixel_values = args_[1] if len(args_) > 1 else kwargs.get("pixel_values")
        before_rope_deltas = language_model._rope_deltas
        result = original_get_input_embeddings(*args_, **kwargs)
        after_rope_deltas = language_model._rope_deltas
        if pixel_values is not None and input_ids is not None:
            input_id_list = input_ids.squeeze(0).tolist()
            if input_id_list == extended_prompt_inputs.prompt_input_ids:
                embedding_probe["before_was_none"] = before_rope_deltas is None
                embedding_probe["after_is_not_none"] = after_rope_deltas is not None
                if after_rope_deltas is not None:
                    mx.eval(after_rope_deltas)
                    embedding_probe["shape"] = tuple(after_rope_deltas.shape)
        return result

    coordinator.restore = restore_wrapper
    model_kit.model.get_input_embeddings = get_input_embeddings_wrapper
    extended_trace = PromptProgressTrace()
    try:
        extended_result = run_prepared_request(
            model_kit,
            extended_prepared,
            prompt_progress_reporter=TracePromptProgressReporter(extended_trace),
        )
    finally:
        coordinator.restore = original_restore
        model_kit.model.get_input_embeddings = original_get_input_embeddings

    if extended_result.stop_reason is None:
        raise RuntimeError("Qwen rope restored request completed without a stop reason")
    if restore_probe.get("cached_prefix_len") != expected_restored_tokens:
        raise RuntimeError(
            "Qwen rope restore did not use the expected disk prefix: "
            f"{restore_probe.get('cached_prefix_len')} != {expected_restored_tokens}"
        )
    if not restore_probe.get("rope_deltas_is_none"):
        raise RuntimeError("Qwen disk restore unexpectedly carried rope_deltas")
    if not embedding_probe.get("before_was_none"):
        raise RuntimeError("Qwen rope state was not cleared before recompute")
    if not embedding_probe.get("after_is_not_none"):
        raise RuntimeError("Qwen get_input_embeddings did not recompute rope_deltas")
    if extended_trace.begin_prefill_tokens_processed != expected_restored_tokens:
        raise RuntimeError(
            "Qwen rope request did not resume from the expected cached prefix: "
            f"{extended_trace.begin_prefill_tokens_processed} "
            f"!= {expected_restored_tokens}"
        )

    print("QWEN_ROPE_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids))
    print(
        "QWEN_ROPE_EXTENDED_PROMPT_TOKENS",
        len(extended_prompt_inputs.prompt_input_ids),
    )
    print("QWEN_ROPE_RESTORED_TOKENS", expected_restored_tokens)
    print("QWEN_ROPE_RESTORE_DELTAS_NONE", restore_probe["rope_deltas_is_none"])
    print("QWEN_ROPE_RECOMPUTED_SHAPE", embedding_probe["shape"])
    print(
        "QWEN_ROPE_BEGIN",
        extended_trace.begin_cached_tokens,
        extended_trace.begin_total_prompt_tokens,
        extended_trace.begin_prefill_tokens_processed,
    )
    print("QWEN_ROPE_TEXT", repr(truncate_text(extended_result.text)))
    print("QWEN_ROPE_E2E_OK")


def run_eviction_e2e(model_kit, processor, args) -> None:
    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        reusable_boundaries,
        suffix,
    ) = build_multi_prefix_e2e_requests(model_kit, processor)
    prompt_cache_store = model_kit._prompt_cache_store

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    if base_result.stop_reason is None:
        raise RuntimeError("Base e2e eviction request completed without a stop reason")

    two_chunk_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
        image_spans=extended_prompt_inputs.image_spans,
        min_chunk_keys=2,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_two_chunk_boundary = reusable_boundaries[1]
    if restore_cached_prefix_len(two_chunk_match) != expected_two_chunk_boundary:
        raise RuntimeError(
            "E2E eviction setup did not build a two-chunk cache: "
            f"{restore_cached_prefix_len(two_chunk_match)} != {expected_two_chunk_boundary}"
        )

    first_key, second_key = restore_chunk_keys(two_chunk_match)[:2]
    first_size = get_cache_store_chunk_size(prompt_cache_store, first_key)
    second_size = get_cache_store_chunk_size(prompt_cache_store, second_key)
    total_before_eviction = prompt_cache_store._total_bytes
    stale_optional_record_keys = get_stale_optional_record_keys(
        prompt_cache_store,
        two_chunk_match,
    )
    stale_optional_size = sum(
        get_cache_store_record_size(prompt_cache_store, record_key)
        for record_key in stale_optional_record_keys
    )
    expected_one_chunk_boundary = reusable_boundaries[0]
    expected_post_eviction_boundary = (
        expected_two_chunk_boundary
        if stale_optional_record_keys
        else expected_one_chunk_boundary
    )
    default_max_bytes = (
        total_before_eviction - stale_optional_size + 1
        if stale_optional_record_keys
        else first_size + 64
    )
    prompt_cache_store._max_cache_store_bytes = (
        args.cache_max_bytes or default_max_bytes
    )
    prompt_cache_store._evict_if_needed()

    first_exists = cache_store_chunk_records_available(prompt_cache_store, first_key)
    second_exists = cache_store_chunk_records_available(prompt_cache_store, second_key)
    if prompt_cache_store._total_bytes > prompt_cache_store._max_cache_store_bytes:
        raise RuntimeError("E2E eviction did not enforce the byte budget")

    post_eviction_match = find_restore_plan(
        prompt_cache_store,
        extended_prompt_inputs.prompt_input_ids,
        extended_prompt_inputs.image_spans,
    )
    if (
        post_eviction_match is None
        or restore_cached_prefix_len(post_eviction_match)
        != expected_post_eviction_boundary
    ):
        actual = (
            None
            if post_eviction_match is None
            else restore_cached_prefix_len(post_eviction_match)
        )
        raise RuntimeError(
            "E2E eviction preserved the wrong reusable prefix: "
            f"{actual} != {expected_post_eviction_boundary}"
        )
    print_cache_store_stats("EVICTION_E2E_AFTER_EVICT", prompt_cache_store)

    clear_hot_completed_prompt_cache(model_kit)
    extended_trace = PromptProgressTrace()
    extended_result = run_prepared_request(
        model_kit,
        extended_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(extended_trace),
    )
    if extended_result.stop_reason is None:
        raise RuntimeError(
            "Extended e2e eviction request completed without a stop reason"
        )
    if extended_trace.begin_prefill_tokens_processed != expected_post_eviction_boundary:
        raise RuntimeError(
            "E2E eviction request did not restore the expected prefix: "
            f"{extended_trace.begin_prefill_tokens_processed} "
            f"!= {expected_post_eviction_boundary}"
        )

    cache_file_sizes = prompt_cache_store.snapshot_stats().record_sizes
    print("EVICTION_E2E_SUFFIX", repr(suffix))
    print("EVICTION_E2E_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids))
    print(
        "EVICTION_E2E_EXTENDED_PROMPT_TOKENS",
        len(extended_prompt_inputs.prompt_input_ids),
    )
    print("EVICTION_E2E_BOUNDARIES", reusable_boundaries)
    print("EVICTION_E2E_FIRST_SIZE", first_size)
    print("EVICTION_E2E_SECOND_SIZE", second_size)
    print("EVICTION_E2E_STALE_OPTIONAL_RECORDS", len(stale_optional_record_keys))
    print("EVICTION_E2E_STALE_OPTIONAL_SIZE", stale_optional_size)
    print("EVICTION_E2E_MAX_BYTES", prompt_cache_store._max_cache_store_bytes)
    print("EVICTION_E2E_TOTAL_BEFORE_EVICT", total_before_eviction)
    print("EVICTION_E2E_TOTAL_BYTES", prompt_cache_store._total_bytes)
    print("EVICTION_E2E_FIRST_EXISTS", first_exists)
    print("EVICTION_E2E_SECOND_EXISTS", second_exists)
    print(
        "EVICTION_E2E_POST_EVICT_CHUNK_KEYS",
        len(restore_chunk_keys(post_eviction_match)),
    )
    print("EVICTION_E2E_CACHE_FILES", len(cache_file_sizes))
    print("EVICTION_E2E_CACHE_FILE_SIZES", cache_file_sizes)
    print("EVICTION_E2E_RESTORED_TOKENS", expected_post_eviction_boundary)
    print(
        "EVICTION_E2E_BASE_BEGIN",
        base_trace.begin_cached_tokens,
        base_trace.begin_total_prompt_tokens,
        base_trace.begin_prefill_tokens_processed,
    )
    print(
        "EVICTION_E2E_EXTENDED_BEGIN",
        extended_trace.begin_cached_tokens,
        extended_trace.begin_total_prompt_tokens,
        extended_trace.begin_prefill_tokens_processed,
    )
    print("EVICTION_E2E_BASE_TEXT", repr(truncate_text(base_result.text)))
    print("EVICTION_E2E_EXTENDED_TEXT", repr(truncate_text(extended_result.text)))
    print_cache_store_stats("EVICTION_E2E", prompt_cache_store)
    print("EVICTION_E2E_OK")


def run_eviction_stress(model_kit, processor, args) -> None:
    if args.max_seq_nums < 2:
        raise ValueError("eviction-stress mode requires --max-seq-nums >= 2")

    (
        base_prepared,
        extended_prepared,
        base_prompt_inputs,
        extended_prompt_inputs,
        reusable_boundaries,
        suffix,
    ) = build_multi_prefix_e2e_requests(model_kit, processor)
    prompt_cache_store = model_kit._prompt_cache_store

    base_trace = PromptProgressTrace()
    base_result = run_prepared_request(
        model_kit,
        base_prepared,
        prompt_progress_reporter=TracePromptProgressReporter(base_trace),
    )
    if base_result.stop_reason is None:
        raise RuntimeError(
            "Base eviction-stress request completed without a stop reason"
        )

    two_chunk_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=extended_prompt_inputs.prompt_input_ids,
        image_spans=extended_prompt_inputs.image_spans,
        min_chunk_keys=2,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_two_chunk_boundary = reusable_boundaries[1]
    if restore_cached_prefix_len(two_chunk_match) != expected_two_chunk_boundary:
        raise RuntimeError(
            "Eviction-stress setup did not build a two-chunk cache: "
            f"{restore_cached_prefix_len(two_chunk_match)} != {expected_two_chunk_boundary}"
        )

    first_key, second_key = restore_chunk_keys(two_chunk_match)[:2]
    first_size = get_cache_store_chunk_size(prompt_cache_store, first_key)
    second_size = get_cache_store_chunk_size(prompt_cache_store, second_key)
    total_before_eviction = prompt_cache_store._total_bytes
    stale_optional_record_keys = get_stale_optional_record_keys(
        prompt_cache_store,
        two_chunk_match,
    )
    stale_optional_size = sum(
        get_cache_store_record_size(prompt_cache_store, record_key)
        for record_key in stale_optional_record_keys
    )
    default_max_bytes = (
        total_before_eviction - stale_optional_size + 1
        if stale_optional_record_keys
        else first_size + 64
    )
    prompt_cache_store._max_cache_store_bytes = (
        args.cache_max_bytes or default_max_bytes
    )
    prompt_cache_store._evict_if_needed()

    expected_one_chunk_boundary = reusable_boundaries[0]
    post_eviction_match = find_restore_plan(
        prompt_cache_store,
        extended_prompt_inputs.prompt_input_ids,
        extended_prompt_inputs.image_spans,
    )
    expected_post_eviction_boundary = (
        expected_two_chunk_boundary
        if stale_optional_record_keys
        else expected_one_chunk_boundary
    )
    if (
        post_eviction_match is None
        or restore_cached_prefix_len(post_eviction_match)
        != expected_post_eviction_boundary
    ):
        actual = (
            None
            if post_eviction_match is None
            else restore_cached_prefix_len(post_eviction_match)
        )
        raise RuntimeError(
            "Eviction-stress preserved the wrong reusable prefix: "
            f"{actual} != {expected_post_eviction_boundary}"
        )
    print_cache_store_stats("EVICTION_STRESS_AFTER_EVICT", prompt_cache_store)

    clear_hot_completed_prompt_cache(model_kit)
    request_count = args.max_seq_nums
    results: dict[str, RequestResult] = {}
    traces: dict[str, PromptProgressTrace] = {}
    errors: dict[str, Exception] = {}

    def make_prepared(index: int) -> PreparedRequest:
        return PreparedRequest(
            request_id=f"eviction-stress-{index + 1}",
            name="eviction-stress",
            prompt_tokens=list(extended_prepared.prompt_tokens),
            images_b64=extended_prepared.images_b64,
            expected_any=(),
            max_tokens=1,
            top_logprobs=0,
            start_delay_s=0.0,
        )

    prepared_requests = [make_prepared(index) for index in range(request_count)]

    def worker(prepared: PreparedRequest) -> None:
        trace = PromptProgressTrace()
        traces[prepared.request_id] = trace
        try:
            results[prepared.request_id] = run_prepared_request(
                model_kit,
                prepared,
                prompt_progress_reporter=TracePromptProgressReporter(trace),
            )
        except Exception as exc:  # pragma: no cover - exercised live
            errors[prepared.request_id] = exc

    threads = [
        threading.Thread(target=worker, args=(prepared,), daemon=True)
        for prepared in prepared_requests
    ]
    start_time = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=args.join_timeout_s)
    wall_time = time.perf_counter() - start_time

    alive_request_ids = [
        prepared.request_id
        for prepared, thread in zip(prepared_requests, threads)
        if thread.is_alive()
    ]
    if alive_request_ids:
        raise RuntimeError(f"Eviction-stress timed out waiting for {alive_request_ids}")
    if errors:
        raise RuntimeError(f"Eviction-stress encountered errors: {errors!r}")
    if len(results) != request_count:
        raise RuntimeError(
            f"Eviction-stress returned {len(results)} results for {request_count} requests"
        )

    restored_tokens = []
    for prepared in prepared_requests:
        result = results[prepared.request_id]
        trace = traces[prepared.request_id]
        if result.stop_reason is None:
            raise RuntimeError(f"{prepared.request_id}: missing stop reason")
        if trace.begin_prefill_tokens_processed is None:
            raise RuntimeError(f"{prepared.request_id}: missing prompt begin")
        restored = trace.begin_prefill_tokens_processed
        if restored not in {expected_one_chunk_boundary, expected_two_chunk_boundary}:
            raise RuntimeError(
                f"{prepared.request_id}: restored unexpected prefix {restored}"
            )
        restored_tokens.append(restored)

    final_stats = wait_for_cache_store_idle(
        prompt_cache_store,
        timeout_s=args.prefix_wait_timeout_s,
    )
    expected_hit_tokens = sum(restored_tokens)
    if final_stats.hit_tokens < expected_hit_tokens:
        raise RuntimeError(
            "Eviction-stress expected at least "
            f"{expected_hit_tokens} hit tokens, got {final_stats.hit_tokens}"
        )
    if final_stats.evictions < 1:
        raise RuntimeError("Eviction-stress expected at least one eviction")

    cache_file_sizes = prompt_cache_store.snapshot_stats().record_sizes
    print("EVICTION_STRESS_SUFFIX", repr(suffix))
    print(
        "EVICTION_STRESS_BASE_PROMPT_TOKENS", len(base_prompt_inputs.prompt_input_ids)
    )
    print(
        "EVICTION_STRESS_EXTENDED_PROMPT_TOKENS",
        len(extended_prompt_inputs.prompt_input_ids),
    )
    print("EVICTION_STRESS_BOUNDARIES", reusable_boundaries)
    print("EVICTION_STRESS_FIRST_SIZE", first_size)
    print("EVICTION_STRESS_SECOND_SIZE", second_size)
    print("EVICTION_STRESS_STALE_OPTIONAL_RECORDS", len(stale_optional_record_keys))
    print("EVICTION_STRESS_STALE_OPTIONAL_SIZE", stale_optional_size)
    print("EVICTION_STRESS_TOTAL_BEFORE_EVICT", total_before_eviction)
    print(
        "EVICTION_STRESS_POST_EVICT_CHUNK_KEYS",
        len(restore_chunk_keys(post_eviction_match)),
    )
    print("EVICTION_STRESS_REQUESTS", request_count)
    print("EVICTION_STRESS_WALL_TIME", f"{wall_time:.2f}s")
    print("EVICTION_STRESS_RESTORED_TOKENS", restored_tokens)
    print("EVICTION_STRESS_CACHE_FILES", len(cache_file_sizes))
    print("EVICTION_STRESS_CACHE_FILE_SIZES", cache_file_sizes)
    print("EVICTION_STRESS_BASE_TEXT", repr(truncate_text(base_result.text)))
    print_cache_store_stats("EVICTION_STRESS", prompt_cache_store)
    print("EVICTION_STRESS_OK")


def run_cross_thread_restore(
    model_kit,
    processor,
    *,
    primary_image_b64: str,
    args,
) -> None:
    prepared = prepare_request(
        model_kit,
        processor,
        RequestSpec(
            name="thread-hop-base",
            prompt=args.prompt,
            expected_any=("bird", "toucan"),
            image_b64=primary_image_b64,
            max_tokens=args.max_tokens,
        ),
        request_id="thread-hop-base",
    )
    prompt_inputs = prepare_prompt_inputs(model_kit, prepared)
    prompt_cache_store = model_kit._prompt_cache_store

    base_result = run_prepared_request(model_kit, prepared)
    validate_result(prepared, base_result)

    prefix_match = wait_for_prefix_snapshot(
        model_kit,
        prompt_input_ids=prompt_inputs.prompt_input_ids,
        image_spans=prompt_inputs.image_spans,
        timeout_s=args.prefix_wait_timeout_s,
    )
    load_state_holder = {}
    load_error_holder = {}
    load_thread_name = "cross-thread-loader"

    def load_on_background_thread() -> None:
        deadline = time.perf_counter() + args.prefix_wait_timeout_s
        while time.perf_counter() < deadline:
            try:
                stored_state = load_restore_plan(prompt_cache_store, prefix_match)
            except Exception as exc:  # pragma: no cover - exercised live
                load_error_holder["error"] = exc
                return
            if stored_state is not None:
                load_state_holder["state"] = stored_state
                return
            time.sleep(0.05)
        load_error_holder["error"] = RuntimeError(
            "Timed out loading prompt cache store snapshot on the background thread"
        )

    loader_thread = threading.Thread(
        target=load_on_background_thread,
        name=load_thread_name,
        daemon=True,
    )
    loader_thread.start()
    loader_thread.join(timeout=args.prefix_wait_timeout_s + 1.0)
    if loader_thread.is_alive():
        raise RuntimeError("Background cache store-load thread did not finish")
    if "error" in load_error_holder:
        raise load_error_holder["error"]
    stored_state = load_state_holder.get("state")
    if stored_state is None:
        raise RuntimeError("Background cache store-load thread did not return a state")

    prompt_cache_coordinator = getattr(model_kit, "_prompt_cache_coordinator", None)
    if prompt_cache_coordinator is None:
        raise RuntimeError("cross-thread-restore mode requires the cache coordinator")

    original_restore = prompt_cache_coordinator.restore
    override_state = {"used": False}

    def restore_override(
        *,
        prompt_input_ids: list[int],
        image_spans: list,
    ):
        if (
            override_state["used"]
            or prompt_input_ids != prompt_inputs.prompt_input_ids
            or image_spans != prompt_inputs.image_spans
        ):
            return original_restore(
                prompt_input_ids=prompt_input_ids,
                image_spans=image_spans,
            )

        if not can_trim_prompt_cache(stored_state.prompt_cache):
            raise RuntimeError("Background-loaded prompt cache is not trimmable")
        trim_count = stored_state.cached_prefix_len - restore_cached_prefix_len(
            prefix_match
        )
        trimmed_tokens = trim_prompt_cache(stored_state.prompt_cache, trim_count)
        if trimmed_tokens != trim_count:
            raise RuntimeError(
                f"Expected to trim {trim_count} tokens from background-loaded prompt cache, got {trimmed_tokens}"
            )

        override_state["used"] = True
        return RestoredPromptCache(
            cached_prefix_len=restore_cached_prefix_len(prefix_match),
            prompt_cache=stored_state.prompt_cache,
            rope_deltas=None,
        )

    prompt_cache_coordinator.restore = restore_override
    trace = PromptProgressTrace()
    try:
        hopped_result = run_prepared_request(
            model_kit,
            prepared,
            prompt_progress_reporter=TracePromptProgressReporter(trace),
        )
    finally:
        prompt_cache_coordinator.restore = original_restore

    validate_result(prepared, hopped_result)
    if not override_state["used"]:
        raise RuntimeError(
            "Generation thread did not consume the background-loaded state"
        )
    if trace.begin_prefill_tokens_processed is None:
        raise RuntimeError("Cross-thread request did not emit a begin progress event")
    expected_progress = restore_cached_prefix_len(prefix_match)
    if trace.begin_prefill_tokens_processed != expected_progress:
        raise RuntimeError(
            "Cross-thread restore did not resume from the expected chunk prefix: "
            f"{trace.begin_prefill_tokens_processed} != {expected_progress}"
        )

    print("THREAD_HOP_LOAD_THREAD", load_thread_name)
    print("THREAD_HOP_PROMPT_TOKENS", len(prompt_inputs.prompt_input_ids))
    print("THREAD_HOP_RESTORED_TOKENS", expected_progress)
    print(
        "THREAD_HOP_BEGIN",
        trace.begin_cached_tokens,
        trace.begin_total_prompt_tokens,
        trace.begin_prefill_tokens_processed,
    )
    print("THREAD_HOP_TEXT", repr(truncate_text(hopped_result.text)))
    print("THREAD_HOP_OK")


def main():
    args = parse_args()
    if (
        args.mode
        in {
            "arrays-cache-e2e",
            "arrays-cache-branch-e2e",
            "qwen-rope-e2e",
        }
        and args.model == DEFAULT_MODEL_PATH
    ):
        args.model = DEFAULT_QWEN35_MODEL_PATH

    model_path = args.model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    secondary_image_path = args.secondary_image.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if (
        args.mode
        in {
            "stress",
            "prefix",
            "prefix-restart",
        }
        and not secondary_image_path.exists()
    ):
        raise FileNotFoundError(f"Secondary image not found: {secondary_image_path}")

    if args.cache_max_bytes is not None:
        print("CACHE_MAX_BYTES", args.cache_max_bytes)

    if args.mode == "multi-prefix":
        run_multi_prefix(args)
        return
    if args.mode == "rotating-suffix":
        run_rotating_suffix(args)
        return
    if args.mode == "eviction":
        run_eviction(args)
        return

    primary_image_b64 = encode_image(image_path)
    secondary_image_b64 = encode_image(secondary_image_path)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model_kit = load_model(
        model_path=model_path,
        max_kv_size=args.cache_max_bytes,
        max_seq_nums=args.max_seq_nums,
        trust_remote_code=args.trust_remote_code,
    )
    print("MODEL_KIT", type(model_kit).__name__)
    if type(model_kit).__name__ != "BatchedVisionModelKit":
        raise RuntimeError(f"Unexpected backend: {type(model_kit).__name__}")

    try:
        if args.mode == "smoke":
            run_smoke(model_kit, processor, args, primary_image_b64)
        elif args.mode in {"prefix", "prefix-restart"}:
            run_prefix(
                model_kit,
                processor,
                primary_image_b64=primary_image_b64,
                secondary_image_b64=secondary_image_b64,
                args=args,
                # The disk cache is per model load, so cross-process restart
                # reuse is intentionally cold.
                expect_warm_base=False,
            )
        elif args.mode == "cross-thread-restore":
            run_cross_thread_restore(
                model_kit,
                processor,
                primary_image_b64=primary_image_b64,
                args=args,
            )
        elif args.mode == "multi-prefix-e2e":
            run_multi_prefix_e2e(model_kit, processor, args)
        elif args.mode == "hybrid-cache-e2e":
            run_hybrid_cache_e2e(model_kit, processor, args)
        elif args.mode == "arrays-cache-e2e":
            run_hybrid_cache_e2e(
                model_kit,
                processor,
                args,
                required_cache_class="ArraysCache",
            )
            print("ARRAYS_CACHE_E2E_OK")
        elif args.mode == "arrays-cache-branch-e2e":
            run_arrays_cache_branch_e2e(model_kit, processor, args)
        elif args.mode == "qwen-rope-e2e":
            run_qwen_rope_e2e(
                model_kit,
                processor,
                primary_image_b64=primary_image_b64,
                args=args,
            )
        elif args.mode == "eviction-e2e":
            run_eviction_e2e(model_kit, processor, args)
        elif args.mode == "eviction-stress":
            run_eviction_stress(model_kit, processor, args)
        else:
            run_stress(
                model_kit,
                processor,
                primary_image_b64=primary_image_b64,
                secondary_image_b64=secondary_image_b64,
                args=args,
            )
    finally:
        unload(model_kit)


if __name__ == "__main__":
    main()
