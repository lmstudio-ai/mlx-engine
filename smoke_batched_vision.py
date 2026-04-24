import argparse
import base64
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MLX_ENGINE_USE_MLX_VLM_BATCHED_VISION", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import AutoProcessor

from mlx_engine.generate import create_generator, load_model, tokenize, unload

DEFAULT_MODEL_PATH = Path(
    "~/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-MLX-4bit"
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke and stress test the feature-flagged mlx-vlm batched vision path"
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "stress"],
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
    return PreparedRequest(
        request_id=request_id,
        name=spec.name,
        prompt_tokens=tokenize(model_kit, prompt),
        images_b64=[spec.image_b64] if spec.image_b64 is not None else None,
        expected_any=spec.expected_any,
        max_tokens=spec.max_tokens,
        top_logprobs=spec.top_logprobs,
        start_delay_s=spec.start_delay_s,
    )


def run_prepared_request(model_kit, prepared: PreparedRequest) -> RequestResult:
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


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    secondary_image_path = args.secondary_image.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.mode == "stress" and not secondary_image_path.exists():
        raise FileNotFoundError(f"Secondary image not found: {secondary_image_path}")

    primary_image_b64 = encode_image(image_path)
    secondary_image_b64 = encode_image(secondary_image_path)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model_kit = load_model(
        model_path=model_path,
        max_seq_nums=args.max_seq_nums,
        trust_remote_code=args.trust_remote_code,
    )
    print("MODEL_KIT", type(model_kit).__name__)
    if type(model_kit).__name__ != "BatchedVisionModelKit":
        raise RuntimeError(f"Unexpected backend: {type(model_kit).__name__}")

    try:
        if args.mode == "smoke":
            run_smoke(model_kit, processor, args, primary_image_b64)
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
