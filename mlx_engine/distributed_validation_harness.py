import argparse
from dataclasses import dataclass
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)
protocol_write_lock = threading.Lock()

default_prompt_a = """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
"""
default_prompt_b = """<|im_start|>user
Explain how photosynthesis works in plants.<|im_end|>
<|im_start|>assistant
"""


@dataclass
class HarnessRequest:
    request_id: str
    prompt: str
    expected_substring: str | None
    forbidden_substring: str | None


@dataclass
class HarnessResult:
    elapsed_seconds: float
    finish_reason: str
    request_id: str
    text: str


def write_protocol_message(message: dict[str, Any]) -> None:
    with protocol_write_lock:
        sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def finish_reason_from_stop_condition(stop_condition: Any) -> Optional[str]:
    if stop_condition is None:
        return None
    if stop_condition.stop_reason == "token_limit":
        return "length"
    if stop_condition.stop_reason == "user_cancelled":
        return "cancelled"
    return "stop"


def sampling_request(seed: int | None) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "topP": None,
        "topK": None,
        "minP": None,
        "repetitionPenalty": None,
        "seed": seed,
    }


def build_request_specs(args: argparse.Namespace) -> list[HarnessRequest]:
    if args.scenario == "single":
        return [
            HarnessRequest(
                request_id="single-a",
                prompt=args.prompt_a,
                expected_substring=None
                if args.skip_output_substring_check
                else args.expect_a,
                forbidden_substring=None
                if args.skip_output_substring_check
                else args.forbid_a,
            )
        ]

    return [
        HarnessRequest(
            request_id="concurrent-a",
            prompt=args.prompt_a,
            expected_substring=None if args.skip_output_substring_check else args.expect_a,
            forbidden_substring=None if args.skip_output_substring_check else args.forbid_a,
        ),
        HarnessRequest(
            request_id="concurrent-b",
            prompt=args.prompt_b,
            expected_substring=None if args.skip_output_substring_check else args.expect_b,
            forbidden_substring=None if args.skip_output_substring_check else args.forbid_b,
        ),
    ]


def validate_request_results(
    requests: list[HarnessRequest], results: dict[str, HarnessResult]
) -> None:
    for request in requests:
        result = results.get(request.request_id)
        if result is None:
            raise AssertionError(f"Missing result for request {request.request_id}")
        if len(result.text.strip()) == 0:
            raise AssertionError(f"Request {request.request_id} produced empty text")
        if request.expected_substring is None:
            continue
        if request.expected_substring.lower() not in result.text.lower():
            raise AssertionError(
                f"Request {request.request_id} output did not contain "
                f"{request.expected_substring!r}: {result.text!r}"
            )
        if (
            request.forbidden_substring is not None
            and request.forbidden_substring.lower() in result.text.lower()
        ):
            raise AssertionError(
                f"Request {request.request_id} output contained cross-wired "
                f"{request.forbidden_substring!r}: {result.text!r}"
            )


def run_generation_request(
    *,
    max_tokens: int,
    model_kit: Any,
    rank: int,
    request: HarnessRequest,
    seed: int | None,
    stop_strings: list[str] | None,
) -> HarnessResult:
    from mlx_engine.distributed_rank import (
        broadcast_generation_request,
        should_broadcast_generation_request,
    )
    from mlx_engine.generate import create_generator

    started_at = time.monotonic()
    prompt_tokens = model_kit.tokenize(request.prompt)
    if should_broadcast_generation_request(model_kit):
        rank_request = broadcast_generation_request(
            rank=rank,
            request_id=request.request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stop=stop_strings,
            sampling=sampling_request(seed),
        )
    else:
        rank_request = {
            "requestId": request.request_id,
            "promptTokens": prompt_tokens,
            "maxTokens": max_tokens,
            "stop": stop_strings,
            "sampling": sampling_request(seed),
        }
    text_parts: list[str] = []
    finish_reason = None

    generator = create_generator(
        model_kit,
        rank_request["promptTokens"],
        max_tokens=rank_request["maxTokens"],
        stop_strings=rank_request["stop"],
        temp=rank_request["sampling"]["temperature"],
        top_p=rank_request["sampling"]["topP"],
        top_k=rank_request["sampling"]["topK"],
        min_p=rank_request["sampling"]["minP"],
        repetition_penalty=rank_request["sampling"]["repetitionPenalty"],
        seed=rank_request["sampling"]["seed"],
        request_id=rank_request["requestId"],
    )

    for generation_result in generator:
        text_parts.append(generation_result.text)
        next_finish_reason = finish_reason_from_stop_condition(
            generation_result.stop_condition
        )
        if next_finish_reason is not None:
            finish_reason = next_finish_reason

    if finish_reason is None:
        finish_reason = "stop"

    return HarnessResult(
        elapsed_seconds=time.monotonic() - started_at,
        finish_reason=finish_reason,
        request_id=request.request_id,
        text="".join(text_parts),
    )


def run_requests(
    *,
    args: argparse.Namespace,
    model_kit: Any,
    rank: int,
    requests: list[HarnessRequest],
) -> dict[str, HarnessResult]:
    results: dict[str, HarnessResult] = {}
    errors: list[Exception] = []
    results_lock = threading.Lock()

    def run_request(request: HarnessRequest) -> None:
        write_protocol_message(
            {
                "type": "harness-request-started",
                "requestId": request.request_id,
            }
        )
        try:
            result = run_generation_request(
                max_tokens=args.max_tokens,
                model_kit=model_kit,
                rank=rank,
                request=request,
                seed=args.seed,
                stop_strings=args.stop_strings,
            )
            with results_lock:
                results[request.request_id] = result
            write_protocol_message(
                {
                    "type": "harness-request-complete",
                    "elapsedSeconds": result.elapsed_seconds,
                    "finishReason": result.finish_reason,
                    "requestId": result.request_id,
                    "text": result.text,
                }
            )
        except Exception as caught_error:
            with results_lock:
                errors.append(caught_error)
            write_protocol_message(
                {
                    "type": "harness-request-error",
                    "message": str(caught_error),
                    "requestId": request.request_id,
                }
            )

    threads = [
        threading.Thread(
            target=run_request,
            args=(request,),
            daemon=True,
            name=f"mlx-distributed-harness-{request.request_id}",
        )
        for request in requests
    ]

    for thread in threads:
        thread.start()

    deadline = time.monotonic() + args.timeout_seconds
    for thread in threads:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break
        thread.join(timeout=remaining_seconds)

    alive_threads = [thread.name for thread in threads if thread.is_alive()]
    if len(alive_threads) > 0:
        raise TimeoutError(
            "Timed out waiting for distributed harness requests: "
            + ", ".join(alive_threads)
        )
    if len(errors) > 0:
        raise RuntimeError(f"{len(errors)} distributed harness request(s) failed")

    return results


def run_rank(args: argparse.Namespace) -> int:
    import mlx.core as mx

    group = mx.distributed.init()
    rank = group.rank()
    size = group.size()
    write_protocol_message(
        {
            "type": "harness-rank-initialized",
            "rank": rank,
            "size": size,
        }
    )

    from mlx_engine import load_model, unload
    from mlx_engine.distributed_rank import (
        run_worker_loop,
        shutdown_workers,
        uses_distributed_batching,
    )

    model_kit = None
    should_shutdown_workers = False
    try:
        model_kit = load_model(
            args.model,
            max_kv_size=args.max_kv_size,
            max_seq_nums=args.max_seq_nums,
            trust_remote_code=args.trust_remote_code,
            prefill_step_size=args.prefill_step_size,
            distributed=True,
            distributed_group=group,
        )
        if rank != 0:
            run_worker_loop(rank, model_kit)
            return 0

        should_shutdown_workers = True
        requests = build_request_specs(args)
        write_protocol_message(
            {
                "type": "harness-scenario-started",
                "maxSeqNums": args.max_seq_nums,
                "requestIds": [request.request_id for request in requests],
                "scenario": args.scenario,
            }
        )
        started_at = time.monotonic()
        results = run_requests(
            args=args,
            model_kit=model_kit,
            rank=rank,
            requests=requests,
        )
        validate_request_results(requests, results)
        write_protocol_message(
            {
                "type": "harness-scenario-complete",
                "elapsedSeconds": time.monotonic() - started_at,
                "requestIds": [request.request_id for request in requests],
                "scenario": args.scenario,
            }
        )
        return 0
    except Exception as caught_error:
        if isinstance(caught_error, TimeoutError):
            should_shutdown_workers = False
        if rank == 0:
            write_protocol_message(
                {
                    "type": "harness-error",
                    "message": str(caught_error),
                    "rank": rank,
                }
            )
        logger.exception("Distributed validation harness failed on rank %s", rank)
        return 1
    finally:
        if rank == 0 and should_shutdown_workers:
            if model_kit is not None and uses_distributed_batching(model_kit):
                model_kit.shutdown()
            else:
                shutdown_workers(rank)
        if model_kit is not None:
            unload(model_kit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX distributed continuous-batching validation harness."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scenario",
        choices=["single", "concurrent"],
        default="concurrent",
    )
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--max-seq-nums", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-a", default=default_prompt_a)
    parser.add_argument("--prompt-b", default=default_prompt_b)
    parser.add_argument("--expect-a", default="paris")
    parser.add_argument("--expect-b", default="chlorophyll")
    parser.add_argument("--forbid-a", default="chlorophyll")
    parser.add_argument("--forbid-b", default="paris")
    parser.add_argument("--skip-output-substring-check", action="store_true")
    parser.add_argument("--stop-strings", nargs="*", default=None)
    args = parser.parse_args()
    if args.max_seq_nums is None:
        args.max_seq_nums = 1 if args.scenario == "single" else 2
    return args


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    exit_code = run_rank(parse_args())
    if exit_code != 0:
        os._exit(exit_code)


if __name__ == "__main__":
    main()
