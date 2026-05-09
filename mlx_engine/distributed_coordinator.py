import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)

distributed_init_started_at_env = "MLX_ENGINE_DISTRIBUTED_COORDINATOR_INIT_STARTED_AT"
source_checkout_runtime_path_segments = (
    "/electron/vendor/llm-engine/build/",
    "/electron/vendor/llm-engine/src/",
)


def normalized_runtime_path(path_value: str) -> str:
    return path_value.replace("\\", "/")


def assert_not_source_checkout_runtime() -> None:
    python_executable = normalized_runtime_path(sys.executable)
    for source_path_segment in source_checkout_runtime_path_segments:
        if source_path_segment in python_executable:
            raise RuntimeError(
                "MLX distributed packaged rank is running from source-checkout "
                f"Python {sys.executable}. Rebuild or stage the packaged MLX "
                "runtime so ranks use the installed Amphibian Python."
            )


def init_distributed_with_retry(timeout_seconds: float):
    started_at = float(os.environ.get(distributed_init_started_at_env, time.monotonic()))
    os.environ[distributed_init_started_at_env] = str(started_at)

    try:
        logger.info("Importing mlx.core before distributed coordinator init")
        import mlx.core as mx

        logger.info("Calling distributed coordinator init")
        group = mx.distributed.init()
        logger.info(
            "Distributed coordinator init completed rank %s/%s",
            group.rank(),
            group.size(),
        )
        return group
    except RuntimeError as error:
        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds >= timeout_seconds:
            raise
        logger.info(
            "Distributed init failed while waiting for worker ranks after %.1fs: %s",
            elapsed_seconds,
            error,
        )
        time.sleep(min(2.0, max(0.0, timeout_seconds - elapsed_seconds)))
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-m",
                "mlx_engine.distributed_coordinator",
                *sys.argv[1:],
            ],
        )
        raise


def run_collective_smoke(rank: int, size: int) -> None:
    import mlx.core as mx

    logger.info("Running coordinator CPU collective smoke rank %s/%s", rank, size)
    cpu_result = mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu)
    mx.eval(cpu_result)
    logger.info(
        "Coordinator CPU collective smoke completed rank %s/%s result=%s",
        rank,
        size,
        cpu_result.item(),
    )

    logger.info(
        "Running coordinator default-stream collective smoke rank %s/%s",
        rank,
        size,
    )
    default_result = mx.distributed.all_sum(mx.array(1.0))
    mx.eval(default_result)
    logger.info(
        "Coordinator default-stream collective smoke completed rank %s/%s result=%s",
        rank,
        size,
        default_result.item(),
    )


def write_protocol_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def optional_number(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number or null")
    return float(value)


def optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer or null")
    return value


def optional_string_list(value: Any, label: str) -> Optional[list[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a string list or null")
    result = []
    for item in value:
        if not isinstance(item, str) or len(item) == 0:
            raise ValueError(f"{label} must only contain non-empty strings")
        result.append(item)
    return result


def validate_sampling(value: Any) -> dict[str, Any]:
    sampling = require_object(value, "sampling")
    return {
        "temperature": optional_number(sampling.get("temperature"), "sampling.temperature"),
        "topP": optional_number(sampling.get("topP"), "sampling.topP"),
        "topK": optional_int(sampling.get("topK"), "sampling.topK"),
        "minP": optional_number(sampling.get("minP"), "sampling.minP"),
        "repetitionPenalty": optional_number(
            sampling.get("repetitionPenalty"), "sampling.repetitionPenalty"
        ),
        "seed": optional_int(sampling.get("seed"), "sampling.seed"),
    }


def validate_predict_command(value: Any) -> dict[str, Any]:
    command = require_object(value, "coordinator command")
    command_type = require_string(command.get("type"), "coordinator command type")
    if command_type != "predict":
        raise ValueError(f"Unsupported coordinator command type {command_type}")

    return {
        "type": "predict",
        "requestId": require_string(command.get("requestId"), "requestId"),
        "prompt": require_string(command.get("prompt"), "prompt"),
        "maxTokens": require_positive_int(command.get("maxTokens"), "maxTokens"),
        "stop": optional_string_list(command.get("stop"), "stop"),
        "sampling": validate_sampling(command.get("sampling")),
    }


def parse_command(line: str) -> dict[str, Any]:
    value = json.loads(line)
    command = require_object(value, "coordinator command")
    command_type = require_string(command.get("type"), "coordinator command type")
    if command_type == "shutdown":
        return {"type": "shutdown"}
    return validate_predict_command(command)


def finish_reason_from_stop_condition(stop_condition) -> Optional[str]:
    if stop_condition is None:
        return None
    if stop_condition.stop_reason == "token_limit":
        return "length"
    if stop_condition.stop_reason == "user_cancelled":
        return "cancelled"
    return "stop"


def run_generation_request(model_kit, request: dict[str, Any]) -> tuple[str, str]:
    from mlx_engine.generate import create_generator

    text_parts = []
    finish_reason = None
    sampling = request["sampling"]
    generator = create_generator(
        model_kit,
        request["promptTokens"],
        max_tokens=request["maxTokens"],
        stop_strings=request["stop"],
        temp=sampling["temperature"],
        top_p=sampling["topP"],
        top_k=sampling["topK"],
        min_p=sampling["minP"],
        repetition_penalty=sampling["repetitionPenalty"],
        seed=sampling["seed"],
        request_id=request["requestId"],
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
    return "".join(text_parts), finish_reason


def handle_predict_command(rank: int, model_kit, command: dict[str, Any]) -> None:
    from mlx_engine.distributed_rank import broadcast_generation_request

    prompt_tokens = model_kit.tokenize(command["prompt"])
    request = broadcast_generation_request(
        rank=rank,
        request_id=command["requestId"],
        prompt_tokens=prompt_tokens,
        max_tokens=command["maxTokens"],
        stop=command["stop"],
        sampling=command["sampling"],
    )
    text, finish_reason = run_generation_request(model_kit, request)
    write_protocol_message(
        {
            "type": "prediction-complete",
            "requestId": request["requestId"],
            "text": text,
            "finishReason": finish_reason,
        }
    )


def run_command_loop(rank: int, model_kit) -> None:
    for line in sys.stdin:
        stripped_line = line.strip()
        if len(stripped_line) == 0:
            continue
        try:
            command = parse_command(stripped_line)
            if command["type"] == "shutdown":
                from mlx_engine.distributed_rank import shutdown_workers

                shutdown_workers(rank)
                return
            handle_predict_command(rank, model_kit, command)
        except Exception as error:
            logger.exception("Distributed coordinator command failed")
            write_protocol_message({"type": "error", "message": str(error)})
            logging.shutdown()
            os._exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Packaged mlx-engine distributed coordinator rank loop."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--distributed-init-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--init-smoke-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    assert_not_source_checkout_runtime()

    group = init_distributed_with_retry(args.distributed_init_timeout_seconds)
    rank = group.rank()
    size = group.size()
    if rank != 0:
        raise RuntimeError("Packaged distributed coordinator must run as rank 0.")
    if args.init_smoke_only:
        run_collective_smoke(rank, size)
        logger.info("Packaged distributed coordinator init smoke completed rank %s/%s", rank, size)
        write_protocol_message({"type": "init-smoke-complete", "rank": rank, "size": size})
        return

    from mlx_engine import load_model, unload

    logger.info(
        "Starting packaged distributed coordinator rank %s/%s model=%s max_kv_size=%s prefill_step_size=%s",
        rank,
        size,
        args.model,
        args.max_kv_size,
        args.prefill_step_size,
    )
    logger.info("Coordinator rank %s calling mlx_engine.load_model(distributed=True)", rank)
    model_kit = load_model(
        args.model,
        max_kv_size=args.max_kv_size,
        max_seq_nums=1,
        trust_remote_code=args.trust_remote_code,
        prefill_step_size=args.prefill_step_size,
        distributed=True,
        distributed_group=group,
    )
    logger.info("Coordinator rank %s mlx_engine.load_model returned", rank)
    try:
        write_protocol_message({"type": "ready", "rank": rank, "size": size})
        logger.info("Coordinator rank %s entering command loop", rank)
        run_command_loop(rank, model_kit)
    finally:
        logger.info("Coordinator rank %s unloading model kit", rank)
        unload(model_kit)


if __name__ == "__main__":
    main()
