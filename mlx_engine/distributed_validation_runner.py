from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


default_prompt_a = """<|im_start|>user
Write a short paragraph about the Eiffel Tower in Paris.<|im_end|>
<|im_start|>assistant
"""
default_prompt_b = """<|im_start|>user
Explain how photosynthesis works in plants.<|im_end|>
<|im_start|>assistant
"""


@dataclass
class ValidationRequest:
    expected_substring: str | None
    forbidden_substring: str | None
    prompt: str
    request_id: str


@dataclass
class ValidationResult:
    finish_reason: str
    request_id: str
    text: str


@dataclass
class OutputState:
    messages: list[dict[str, Any]]
    output_tail: deque[str]
    rank_exit_warning: str | None = None


def stream_lines(
    *,
    condition: threading.Condition,
    output_state: OutputState,
    stream,
    stream_name: str,
) -> None:
    for raw_line in iter(stream.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        output_line = f"[mlx-validation:{stream_name}] {line}"
        with condition:
            if "exited with code" in line:
                output_state.rank_exit_warning = line
            protocol_message = parse_protocol_message(line)
            if protocol_message is not None:
                output_state.messages.append(protocol_message)
            output_state.output_tail.append(output_line)
            condition.notify_all()
        print(output_line, flush=True)


def parse_protocol_message(line: str) -> dict[str, Any] | None:
    if not line.startswith("{"):
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None


def print_tail(output_tail: deque[str]) -> None:
    print("[mlx-validation] Output tail:", file=sys.stderr, flush=True)
    for line in output_tail:
        print(line, file=sys.stderr, flush=True)


def resolve_default_python() -> str:
    return sys.executable


def resolve_default_pythonpath() -> str:
    return str(Path(__file__).resolve().parents[1])


def sampling_command(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "topP": None,
        "topK": None,
        "minP": None,
        "repetitionPenalty": None,
        "seed": args.seed,
    }


def build_request_specs(args: argparse.Namespace) -> list[ValidationRequest]:
    if args.scenario == "single":
        return [
            ValidationRequest(
                expected_substring=None
                if args.skip_output_substring_check
                else args.expect_a,
                forbidden_substring=None
                if args.skip_output_substring_check
                else args.forbid_a,
                prompt=args.prompt_a,
                request_id="single-a",
            )
        ]
    return [
        ValidationRequest(
            expected_substring=None if args.skip_output_substring_check else args.expect_a,
            forbidden_substring=None if args.skip_output_substring_check else args.forbid_a,
            prompt=args.prompt_a,
            request_id="concurrent-a",
        ),
        ValidationRequest(
            expected_substring=None if args.skip_output_substring_check else args.expect_b,
            forbidden_substring=None if args.skip_output_substring_check else args.forbid_b,
            prompt=args.prompt_b,
            request_id="concurrent-b",
        ),
    ]


def coordinator_predict_request(
    args: argparse.Namespace,
    request: ValidationRequest,
) -> dict[str, Any]:
    return {
        "type": "predict",
        "requestId": request.request_id,
        "prompt": request.prompt,
        "maxTokens": args.max_tokens,
        "stop": args.stop_strings,
        "sampling": sampling_command(args),
    }


def coordinator_command(
    args: argparse.Namespace,
    requests: list[ValidationRequest],
) -> dict[str, Any]:
    if args.scenario == "single":
        return coordinator_predict_request(args, requests[0])
    return {
        "type": "predict-concurrent",
        "requests": [
            coordinator_predict_request(args, request) for request in requests
        ],
        "timeoutSeconds": args.timeout_seconds,
    }


def coordinator_module_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--model",
        args.model,
        "--max-kv-size",
        str(args.max_kv_size),
        "--distributed-init-timeout-seconds",
        str(args.distributed_init_timeout_seconds),
        "--max-seq-nums",
        str(args.max_seq_nums),
    ]
    if args.prefill_step_size is not None:
        result.extend(["--prefill-step-size", str(args.prefill_step_size)])
    if args.trust_remote_code:
        result.append("--trust-remote-code")
    return result


def start_stream_threads(
    *,
    condition: threading.Condition,
    output_state: OutputState,
    process: subprocess.Popen[bytes],
) -> list[threading.Thread]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to capture mlx.launch output streams")
    threads = [
        threading.Thread(
            target=stream_lines,
            kwargs={
                "condition": condition,
                "output_state": output_state,
                "stream": process.stdout,
                "stream_name": "stdout",
            },
            daemon=True,
        ),
        threading.Thread(
            target=stream_lines,
            kwargs={
                "condition": condition,
                "output_state": output_state,
                "stream": process.stderr,
                "stream_name": "stderr",
            },
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    return threads


def protocol_error_message(message: dict[str, Any]) -> str | None:
    message_type = message.get("type")
    if message_type in {"error", "harness-error", "prediction-error"}:
        raw_message = message.get("message")
        if isinstance(raw_message, str):
            return raw_message
        return f"Protocol message reported {message_type}"
    return None


def wait_for_message(
    *,
    condition: threading.Condition,
    output_state: OutputState,
    predicate: Callable[[dict[str, Any]], bool],
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        with condition:
            for message_index, message in enumerate(output_state.messages):
                error_message = protocol_error_message(message)
                if error_message is not None:
                    output_state.messages.pop(message_index)
                    raise RuntimeError(error_message)
                if predicate(message):
                    return output_state.messages.pop(message_index)
            if output_state.rank_exit_warning is not None:
                raise RuntimeError(output_state.rank_exit_warning)

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"Timed out waiting for MLX validation protocol message after {timeout_seconds}s"
                )
            condition.wait(timeout=min(0.1, remaining_seconds))

        if process.poll() is not None:
            with condition:
                for message_index, message in enumerate(output_state.messages):
                    error_message = protocol_error_message(message)
                    if error_message is not None:
                        output_state.messages.pop(message_index)
                        raise RuntimeError(error_message)
                    if predicate(message):
                        return output_state.messages.pop(message_index)
            raise RuntimeError(f"mlx.launch exited with code {process.returncode}")


def validate_results(
    *,
    requests: list[ValidationRequest],
    results: dict[str, ValidationResult],
) -> None:
    for request in requests:
        result = results.get(request.request_id)
        if result is None:
            raise AssertionError(f"Missing result for request {request.request_id}")
        if len(result.text.strip()) == 0:
            raise AssertionError(f"Request {request.request_id} produced empty text")
        if request.expected_substring is not None:
            if request.expected_substring.lower() not in result.text.lower():
                raise AssertionError(
                    f"Request {request.request_id} output did not contain "
                    f"{request.expected_substring!r}: {result.text!r}"
                )
        if request.forbidden_substring is not None:
            if request.forbidden_substring.lower() in result.text.lower():
                raise AssertionError(
                    f"Request {request.request_id} output contained cross-wired "
                    f"{request.forbidden_substring!r}: {result.text!r}"
                )


def result_from_prediction_message(message: dict[str, Any]) -> ValidationResult:
    request_id = message.get("requestId")
    finish_reason = message.get("finishReason")
    text = message.get("text")
    if not isinstance(request_id, str):
        raise RuntimeError(f"Invalid prediction-complete requestId: {message!r}")
    if not isinstance(finish_reason, str):
        raise RuntimeError(f"Invalid prediction-complete finishReason: {message!r}")
    if not isinstance(text, str):
        raise RuntimeError(f"Invalid prediction-complete text: {message!r}")
    return ValidationResult(
        finish_reason=finish_reason,
        request_id=request_id,
        text=text,
    )


def send_command(process: subprocess.Popen[bytes], command: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("mlx.launch stdin is not available")
    process.stdin.write((json.dumps(command, separators=(",", ":")) + "\n").encode())
    process.stdin.flush()


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_coordinator_validation(
    *,
    args: argparse.Namespace,
    command_prefix: list[str],
    environment: dict[str, str],
) -> int:
    if args.ranks < 2:
        raise ValueError("Coordinator validation requires at least two ranks")

    command = [
        *command_prefix,
        "-m",
        "mlx_engine.distributed_validation_rank_entry",
        *coordinator_module_args(args),
    ]
    print("[mlx-validation] Running " + " ".join(command), flush=True)
    condition = threading.Condition()
    output_state = OutputState(messages=[], output_tail=deque(maxlen=args.tail_lines))
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    threads = start_stream_threads(
        condition=condition,
        output_state=output_state,
        process=process,
    )

    should_send_shutdown = False
    try:
        wait_for_message(
            condition=condition,
            output_state=output_state,
            predicate=lambda message: message.get("type") == "ready",
            process=process,
            timeout_seconds=args.launch_timeout_seconds,
        )
        should_send_shutdown = True
        requests = build_request_specs(args)
        send_command(process, coordinator_command(args, requests))

        results: dict[str, ValidationResult] = {}
        while len(results) < len(requests):
            prediction_message = wait_for_message(
                condition=condition,
                output_state=output_state,
                predicate=lambda message: message.get("type") == "prediction-complete",
                process=process,
                timeout_seconds=args.timeout_seconds,
            )
            result = result_from_prediction_message(prediction_message)
            results[result.request_id] = result

        if args.scenario == "concurrent":
            wait_for_message(
                condition=condition,
                output_state=output_state,
                predicate=lambda message: message.get("type")
                == "prediction-batch-complete",
                process=process,
                timeout_seconds=args.timeout_seconds,
            )

        validate_results(requests=requests, results=results)
        return 0
    except Exception as caught_error:
        print(f"[mlx-validation] Failed: {caught_error}", file=sys.stderr, flush=True)
        print_tail(output_state.output_tail)
        return 1
    finally:
        if should_send_shutdown and process.poll() is None:
            try:
                send_command(process, {"type": "shutdown"})
            except Exception:
                pass
        terminate_process(process)
        for thread in threads:
            thread.join(timeout=1.0)


def run_direct_harness_validation(
    *,
    command_prefix: list[str],
    environment: dict[str, str],
    harness_args: Sequence[str],
    launch_timeout_seconds: float,
    tail_lines: int,
) -> int:
    command = [
        *command_prefix,
        "-m",
        "mlx_engine.distributed_validation_harness",
        *harness_args,
    ]
    print("[mlx-validation] Running " + " ".join(command), flush=True)
    condition = threading.Condition()
    output_state = OutputState(messages=[], output_tail=deque(maxlen=tail_lines))
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    threads = start_stream_threads(
        condition=condition,
        output_state=output_state,
        process=process,
    )
    try:
        wait_for_message(
            condition=condition,
            output_state=output_state,
            predicate=lambda message: message.get("type")
            == "harness-scenario-complete",
            process=process,
            timeout_seconds=launch_timeout_seconds,
        )
        return 0
    except Exception as caught_error:
        print(f"[mlx-validation] Failed: {caught_error}", file=sys.stderr, flush=True)
        print_tail(output_state.output_tail)
        return 1
    finally:
        terminate_process(process)
        for thread in threads:
            thread.join(timeout=1.0)


def parse_validation_args(harness_args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX distributed continuous-batching validation options."
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
    parser.add_argument("--distributed-init-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-a", default=default_prompt_a)
    parser.add_argument("--prompt-b", default=default_prompt_b)
    parser.add_argument("--expect-a", default="paris")
    parser.add_argument("--expect-b", default="chlorophyll")
    parser.add_argument("--forbid-a", default="chlorophyll")
    parser.add_argument("--forbid-b", default="paris")
    parser.add_argument("--skip-output-substring-check", action="store_true")
    parser.add_argument("--stop-strings", nargs="*", default=None)
    args = parser.parse_args(harness_args)
    if args.max_seq_nums is None:
        args.max_seq_nums = 1 if args.scenario == "single" else 2
    return args


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MLX distributed validation under mlx.launch."
    )
    parser.add_argument("--launcher", default="mlx.launch")
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--python", default=resolve_default_python())
    parser.add_argument("--pythonpath", default=resolve_default_pythonpath())
    parser.add_argument("--cwd", default=str(Path.cwd()))
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument(
        "--target",
        choices=["coordinator", "direct-harness"],
        default="coordinator",
    )
    parser.add_argument("--launch-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--tail-lines", type=int, default=120)
    args, harness_args = parser.parse_known_args()
    if len(harness_args) > 0 and harness_args[0] == "--":
        harness_args = harness_args[1:]
    if len(harness_args) == 0:
        raise ValueError("Pass validation arguments after --")
    return args, harness_args


def run_validation(args: argparse.Namespace, harness_args: Sequence[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="mlx-distributed-validation-") as temp_dir:
        python_command_name = "mlx-engine-validation-python"
        python_command_path = Path(temp_dir) / python_command_name
        python_command_path.symlink_to(Path(args.python).resolve())

        environment = os.environ.copy()
        environment["PATH"] = temp_dir + os.pathsep + environment.get("PATH", "")

        command_prefix = [
            args.launcher,
            "-n",
            str(args.ranks),
            "--cwd",
            args.cwd,
            "--env",
            f"PYTHONPATH={args.pythonpath}",
            *args.launcher_arg,
            python_command_name,
        ]

        if args.target == "direct-harness":
            return run_direct_harness_validation(
                command_prefix=command_prefix,
                environment=environment,
                harness_args=harness_args,
                launch_timeout_seconds=args.launch_timeout_seconds,
                tail_lines=args.tail_lines,
            )

        validation_args = parse_validation_args(harness_args)
        validation_args.ranks = args.ranks
        validation_args.launch_timeout_seconds = args.launch_timeout_seconds
        validation_args.tail_lines = args.tail_lines
        return run_coordinator_validation(
            args=validation_args,
            command_prefix=command_prefix,
            environment=environment,
        )


def main() -> None:
    args, harness_args = parse_args()
    sys.exit(run_validation(args, harness_args))


if __name__ == "__main__":
    main()
