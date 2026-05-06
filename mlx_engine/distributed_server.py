import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import mlx.core as mx

from mlx_engine.generate import create_generator, load_model, unload


logger = logging.getLogger(__name__)

RANK_PROTOCOL_VERSION = 1
MESSAGE_TYPE_CHAT_COMPLETIONS = "chat.completions"
MESSAGE_TYPE_SHUTDOWN = "shutdown"
HTTP_DEBUG_SERVER_ENV_VAR = "LMSTUDIO_MLX_DISTRIBUTED_HTTP_DEBUG_SERVER"


def require_http_debug_server_opt_in() -> None:
    if os.environ.get(HTTP_DEBUG_SERVER_ENV_VAR) == "1":
        logger.warning(
            "Starting deprecated MLX distributed HTTP debug harness. "
            "Product distributed inference must use the native llm-engine path."
        )
        return

    raise RuntimeError(
        "mlx_engine.distributed_server is a debug/parity harness only. "
        f"Set {HTTP_DEBUG_SERVER_ENV_VAR}=1 only from an explicit debug harness."
    )


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
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
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{label} must be a non-empty token list")
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
        "repetitionContextSize": require_optional_int(
            sampling.get("repetitionContextSize"),
            "sampling.repetitionContextSize",
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

    if message_type != MESSAGE_TYPE_CHAT_COMPLETIONS:
        raise ValueError(f"Unsupported rank message type: {message_type}")

    return {
        "version": RANK_PROTOCOL_VERSION,
        "type": MESSAGE_TYPE_CHAT_COMPLETIONS,
        "requestId": require_string(message.get("requestId"), "requestId"),
        "model": require_string(message.get("model"), "model"),
        "promptTokens": require_token_list(message.get("promptTokens"), "promptTokens"),
        "maxTokens": require_positive_int(message.get("maxTokens"), "maxTokens"),
        "stop": require_optional_string_list(message.get("stop"), "stop"),
        "sampling": validate_sampling(message.get("sampling")),
        "stream": require_bool(message.get("stream"), "stream"),
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


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for content_part_index, content_part in enumerate(content):
            if not isinstance(content_part, dict):
                raise ValueError(
                    f"message.content[{content_part_index}] must be an object"
                )
            content_part_type = content_part.get("type")
            if content_part_type != "text":
                raise ValueError(
                    "Distributed mlx-engine supports only text message content; "
                    f"unsupported content type: {content_part_type!r}"
                )
            text = content_part.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"message.content[{content_part_index}].text must be a string"
                )
            text_parts.append(text)
        return "".join(text_parts)
    if content is None:
        return ""
    raise ValueError("message.content must be a string or an array of text parts")


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("messages must be a list")

    messages = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("messages entries must be objects")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError("message.role must be a string")
        messages.append(
            {
                "role": role,
                "content": normalize_message_content(message.get("content")),
            }
        )
    return messages


def normalize_stop(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) == 0:
            raise ValueError("stop must not be empty")
        return [value]
    if isinstance(value, list):
        stop_strings = []
        for stop_value in value:
            if not isinstance(stop_value, str):
                raise ValueError("stop must only contain strings")
            if len(stop_value) == 0:
                raise ValueError("stop must not contain empty strings")
            stop_strings.append(stop_value)
        return stop_strings
    return None


def reject_unsupported_request_fields(body: dict[str, Any]) -> None:
    unsupported_fields = [
        "adapters",
        "adapter",
        "draft_model",
        "num_draft_tokens",
    ]
    for field_name in unsupported_fields:
        if field_name in body and body[field_name] is not None:
            raise ValueError(
                f"Distributed mlx-engine does not support request field: {field_name}"
            )


def optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) or isinstance(value, float):
        return float(value)
    return None


def optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def max_tokens_from_value(value: Any) -> int:
    if value is None:
        return 256
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_tokens must be a positive integer")
    max_tokens = value
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return max_tokens


def format_prompt(model_kit, body: dict[str, Any], default_template_args: dict[str, Any]) -> list[int]:
    messages = normalize_messages(body.get("messages"))
    template_args = dict(default_template_args)
    request_template_args = body.get("chat_template_kwargs")
    if isinstance(request_template_args, dict):
        template_args.update(request_template_args)

    prompt = model_kit.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_args,
    )
    return model_kit.tokenize(prompt)


def finish_reason_from_stop_condition(stop_condition) -> Optional[str]:
    if stop_condition is None:
        return None
    if stop_condition.stop_reason == "token_limit":
        return "length"
    if stop_condition.stop_reason == "user_cancelled":
        return "cancelled"
    return "stop"


def run_generation_request(model_kit, request: dict[str, Any]):
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
        repetition_context_size=sampling["repetitionContextSize"],
        seed=sampling["seed"],
        request_id=request["requestId"],
    )

    for generation_result in generator:
        finish_reason = finish_reason_from_stop_condition(
            generation_result.stop_condition
        )
        yield generation_result.text, finish_reason
        text_parts.append(generation_result.text)

    if finish_reason is None:
        finish_reason = "stop"
    return "".join(text_parts), finish_reason


def build_generation_request(
    model_kit,
    body: dict[str, Any],
    model_name: str,
    default_template_args: dict[str, Any],
) -> dict[str, Any]:
    reject_unsupported_request_fields(body)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    seed = optional_int(body.get("seed"))
    if seed is None:
        seed = time.time_ns() & (2**32 - 1)
    return validate_rank_message(
        {
            "version": RANK_PROTOCOL_VERSION,
            "type": MESSAGE_TYPE_CHAT_COMPLETIONS,
            "requestId": request_id,
            "model": (
                body.get("model")
                if isinstance(body.get("model"), str)
                else model_name
            ),
            "promptTokens": format_prompt(model_kit, body, default_template_args),
            "maxTokens": max_tokens_from_value(body.get("max_tokens")),
            "stop": normalize_stop(body.get("stop")),
            "sampling": {
                "temperature": optional_float(body.get("temperature")),
                "topP": optional_float(body.get("top_p")),
                "topK": optional_int(body.get("top_k")),
                "minP": optional_float(body.get("min_p")),
                "repetitionPenalty": optional_float(
                    body.get("repetition_penalty")
                ),
                "repetitionContextSize": optional_int(
                    body.get("repetition_context_size")
                ),
                "seed": seed,
            },
            "stream": body.get("stream") is True,
        }
    )


def log_request_event(message: str, request: dict[str, Any]) -> None:
    logger.info(
        "%s requestId=%s stream=%s maxTokens=%s promptTokens=%s",
        message,
        request["requestId"],
        request["stream"],
        request["maxTokens"],
        len(request["promptTokens"]),
    )


class DistributedEngineHandler(BaseHTTPRequestHandler):
    model_kit = None
    model_name = ""
    rank = 0
    default_template_args: dict[str, Any] = {}
    active_distributed_request = False

    def log_message(self, format_string: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format_string % args)

    def send_json(self, status_code: int, body: dict[str, Any]) -> None:
        encoded_body = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

    def send_error_json(self, status_code: int, message: str) -> None:
        self.send_json(status_code, {"error": {"message": message}})

    def send_pre_broadcast_error(self, status_code: int, message: str) -> None:
        try:
            self.send_error_json(status_code, message)
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected before request was broadcast")

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "rank": self.rank})
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "mlx-engine",
                        }
                    ],
                },
            )
            return
        self.send_error_json(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error_json(404, "Not found")
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self.send_error_json(400, "Missing Content-Length")
            return

        try:
            raw_body = self.rfile.read(int(content_length))
            body = json.loads(raw_body.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            request = build_generation_request(
                self.model_kit,
                body,
                self.model_name,
                self.default_template_args,
            )
        except Exception as caught_error:
            logger.exception("Request failed before broadcast")
            self.send_pre_broadcast_error(400, str(caught_error))
            return

        try:
            log_request_event("Broadcasting distributed request", request)
            share_rank_message(self.rank, request)
        except Exception:
            logger.exception("Request broadcast failed")
            exit_after_distributed_error("Request broadcast failed")

        completed_request = False
        DistributedEngineHandler.active_distributed_request = True
        try:
            if request["stream"] is True:
                self.handle_streaming_request(request)
            else:
                self.handle_non_streaming_request(request)
            completed_request = True
        except Exception:
            logger.exception("Request failed after broadcast")
            exit_after_distributed_error("Post-broadcast request handling failed")
        finally:
            if completed_request:
                DistributedEngineHandler.active_distributed_request = False

    def handle_non_streaming_request(self, request: dict[str, Any]) -> None:
        text_parts = []
        finish_reason = "stop"
        try:
            for text, next_finish_reason in run_generation_request(self.model_kit, request):
                text_parts.append(text)
                if next_finish_reason is not None:
                    finish_reason = next_finish_reason
        except Exception:
            logger.exception("Generation failed")
            exit_after_distributed_error("Generation failed")

        try:
            self.send_json(
                200,
                {
                    "id": request["requestId"],
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(text_parts),
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected after non-streaming generation completed")

    def write_sse(self, body: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(body)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def handle_streaming_request(self, request: dict[str, Any]) -> None:
        client_connected = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected before streaming generation began")
            client_connected = False

        finish_reason = "stop"
        try:
            for text, next_finish_reason in run_generation_request(self.model_kit, request):
                if next_finish_reason is not None:
                    finish_reason = next_finish_reason
                if len(text) == 0:
                    continue
                if client_connected:
                    try:
                        self.write_sse(
                            {
                                "id": request["requestId"],
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request["model"],
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    except (BrokenPipeError, ConnectionResetError):
                        logger.warning("Client disconnected during streaming generation; draining request")
                        client_connected = False
        except Exception:
            logger.exception("Generation failed")
            exit_after_distributed_error("Generation failed")

        if client_connected:
            try:
                self.write_sse(
                    {
                        "id": request["requestId"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.warning("Client disconnected after streaming generation completed")


def run_worker_loop(rank: int, model_kit) -> None:
    logger.info("Rank %s waiting for rank 0 requests", rank)
    while True:
        request = share_rank_message(rank, None)
        if request is None:
            continue
        if request.get("type") == MESSAGE_TYPE_SHUTDOWN:
            return
        log_request_event(f"Rank {rank} starting distributed request", request)
        for _text, _finish_reason in run_generation_request(model_kit, request):
            pass
        log_request_event(f"Rank {rank} finished distributed request", request)


def parse_chat_template_args(value: str) -> dict[str, Any]:
    parsed_value = json.loads(value)
    if not isinstance(parsed_value, dict):
        raise ValueError("--chat-template-args must be a JSON object")
    return parsed_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated distributed mlx-engine HTTP debug harness."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template-args", default="{}")
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    require_http_debug_server_opt_in()
    default_template_args = parse_chat_template_args(args.chat_template_args)

    group = mx.distributed.init()
    rank = group.rank()
    logger.info("Starting distributed mlx-engine rank %s/%s", rank, group.size())

    model_kit = load_model(
        args.model,
        max_kv_size=args.max_kv_size,
        max_seq_nums=1,
        trust_remote_code=args.trust_remote_code,
        prefill_step_size=args.prefill_step_size,
        distributed=True,
        distributed_group=group,
    )

    if rank == 0:
        DistributedEngineHandler.model_kit = model_kit
        DistributedEngineHandler.model_name = args.model
        DistributedEngineHandler.rank = rank
        DistributedEngineHandler.default_template_args = default_template_args
        try:
            server = HTTPServer((args.host, args.port), DistributedEngineHandler)
            logger.info("Serving rank 0 on http://%s:%s", args.host, args.port)
            server.serve_forever()
        finally:
            if DistributedEngineHandler.active_distributed_request:
                logger.warning(
                    "Skipping worker shutdown broadcast because a distributed request is active"
                )
            else:
                shutdown_workers(rank)
            unload(model_kit)
    else:
        try:
            run_worker_loop(rank, model_kit)
        finally:
            unload(model_kit)


if __name__ == "__main__":
    main()
