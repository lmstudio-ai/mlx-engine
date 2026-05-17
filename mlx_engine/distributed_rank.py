import json
import logging
import os
import sys
from typing import Any, Optional

import mlx.core as mx

from mlx_engine.generate import create_generator


logger = logging.getLogger(__name__)

RANK_PROTOCOL_VERSION = 1
MESSAGE_TYPE_GENERATE = "generate"
MESSAGE_TYPE_SHUTDOWN = "shutdown"
SOURCE_CHECKOUT_RUNTIME_PATH_SEGMENTS = (
    "/electron/vendor/llm-engine/build/",
    "/electron/vendor/llm-engine/src/",
)


def normalized_runtime_path(path_value: str) -> str:
    return path_value.replace("\\", "/")


def assert_not_source_checkout_runtime() -> None:
    python_executable = normalized_runtime_path(sys.executable)
    for source_path_segment in SOURCE_CHECKOUT_RUNTIME_PATH_SEGMENTS:
        if source_path_segment in python_executable:
            raise RuntimeError(
                "MLX distributed packaged rank is running from source-checkout "
                f"Python {sys.executable}. Rebuild or stage the packaged MLX "
                "runtime so ranks use the installed Amphibian Python."
            )


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def require_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer or null")
    return value


def require_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number or null")
    return float(value)


def require_token_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a token list")
    tokens = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} must only contain non-negative integers")
        tokens.append(token)
    return tokens


def require_optional_string_list(value: Any, label: str) -> Optional[list[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a string list or null")
    strings = []
    for string_value in value:
        if not isinstance(string_value, str):
            raise ValueError(f"{label} must only contain strings")
        if len(string_value) == 0:
            raise ValueError(f"{label} must not contain empty strings")
        strings.append(string_value)
    return strings


def validate_sampling(value: Any) -> dict[str, Any]:
    sampling = require_object(value, "sampling")
    return {
        "temperature": require_optional_float(
            sampling.get("temperature"), "sampling.temperature"
        ),
        "topP": require_optional_float(sampling.get("topP"), "sampling.topP"),
        "topK": require_optional_int(sampling.get("topK"), "sampling.topK"),
        "minP": require_optional_float(sampling.get("minP"), "sampling.minP"),
        "repetitionPenalty": require_optional_float(
            sampling.get("repetitionPenalty"), "sampling.repetitionPenalty"
        ),
        "seed": require_optional_int(sampling.get("seed"), "sampling.seed"),
    }


def validate_rank_message(value: Any) -> dict[str, Any]:
    message = require_object(value, "rank message")
    version = message.get("version")
    if version != RANK_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported rank protocol version {version!r}; expected {RANK_PROTOCOL_VERSION}"
        )

    message_type = require_string(message.get("type"), "rank message type")
    if message_type == MESSAGE_TYPE_SHUTDOWN:
        return {"version": RANK_PROTOCOL_VERSION, "type": MESSAGE_TYPE_SHUTDOWN}

    if message_type != MESSAGE_TYPE_GENERATE:
        raise ValueError(f"Unsupported rank message type: {message_type}")

    return {
        "version": RANK_PROTOCOL_VERSION,
        "type": MESSAGE_TYPE_GENERATE,
        "requestId": require_string(message.get("requestId"), "requestId"),
        "promptTokens": require_token_list(message.get("promptTokens"), "promptTokens"),
        "maxTokens": require_non_negative_int(message.get("maxTokens"), "maxTokens"),
        "stop": require_optional_string_list(message.get("stop"), "stop"),
        "sampling": validate_sampling(message.get("sampling")),
    }


def encode_rank_message(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def decode_rank_message(data: mx.array) -> dict[str, Any]:
    payload = bytes(data.tolist()).decode("utf-8")
    return validate_rank_message(json.loads(payload))


def share_rank_message(
    rank: int, value: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if rank == 0:
        if value is None:
            mx.eval(mx.distributed.all_sum(0))
            return None
        validated_value = validate_rank_message(value)
        data = mx.array(list(encode_rank_message(validated_value)), dtype=mx.uint8)
        mx.eval(mx.distributed.all_sum(data.size))
        mx.eval(mx.distributed.all_sum(data))
        return validated_value

    data_size = mx.distributed.all_sum(0).item()
    if data_size == 0:
        return None
    data = mx.zeros(data_size, dtype=mx.uint8)
    data = mx.distributed.all_sum(data)
    return decode_rank_message(data)


def exit_after_distributed_error(message: str) -> None:
    logger.critical(
        "%s. Exiting this rank to avoid accepting another request while worker ranks may be out of sync.",
        message,
    )
    logging.shutdown()
    os._exit(1)


def log_request_event(message: str, request: dict[str, Any]) -> None:
    logger.info(
        "%s requestId=%s maxTokens=%s promptTokens=%s",
        message,
        request["requestId"],
        request["maxTokens"],
        len(request["promptTokens"]),
    )


def build_generation_request(
    request_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
    stop: Optional[list[str]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    return validate_rank_message(
        {
            "version": RANK_PROTOCOL_VERSION,
            "type": MESSAGE_TYPE_GENERATE,
            "requestId": request_id,
            "promptTokens": prompt_tokens,
            "maxTokens": max_tokens,
            "stop": stop,
            "sampling": sampling,
        }
    )


def broadcast_generation_request(
    rank: int,
    request_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
    stop: Optional[list[str]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    request = build_generation_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        stop=stop,
        sampling=sampling,
    )
    log_request_event("Broadcasting native distributed request", request)
    return share_rank_message(rank, request)


def shutdown_workers(rank: int) -> None:
    try:
        share_rank_message(
            rank,
            {
                "version": RANK_PROTOCOL_VERSION,
                "type": MESSAGE_TYPE_SHUTDOWN,
            },
        )
    except Exception:
        logger.exception("Failed to broadcast worker shutdown")


def uses_distributed_batching(model_kit) -> bool:
    uses_batching = getattr(model_kit, "uses_distributed_batching", None)
    if uses_batching is None:
        return False
    return bool(uses_batching())


def should_broadcast_generation_request(model_kit) -> bool:
    return not uses_distributed_batching(model_kit)


def run_generation_request(model_kit, request: dict[str, Any]) -> None:
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

    for _generation_result in generator:
        pass


def run_worker_loop(rank: int, model_kit) -> None:
    assert_not_source_checkout_runtime()
    if uses_distributed_batching(model_kit):
        logger.info("Native distributed rank %s entering batched worker loop", rank)
        model_kit.run_worker_loop()
        return

    logger.info("Native distributed rank %s waiting for rank 0 requests", rank)
    while True:
        try:
            request = share_rank_message(rank, None)
            if request is None:
                continue
            if request.get("type") == MESSAGE_TYPE_SHUTDOWN:
                return
            log_request_event(f"Rank {rank} starting native distributed request", request)
            run_generation_request(model_kit, request)
            log_request_event(f"Rank {rank} finished native distributed request", request)
        except Exception:
            logger.exception("Native distributed worker request failed")
            exit_after_distributed_error("Native distributed worker request failed")
