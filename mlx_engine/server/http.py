from __future__ import annotations

import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
from typing import Callable, Iterator
import uuid

from pydantic import ValidationError

from mlx_engine import (
    create_generator,
    get_runtime_load_info,
    stop_generation,
    tokenize,
    unload,
)
from mlx_engine.model_kit.batched_vision import BatchedVisionModelKit
from mlx_engine.utils.generation_result import (
    GenerationResult,
    GenerationStopCondition,
)
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

from .chat import (
    ChatGenerationRequest,
    ChatRequestError,
    prepare_chat_generation_request,
)


logger = logging.getLogger(__name__)


_MAX_REQUEST_BODY_MIB = 64
_MAX_REQUEST_BODY_BYTES = _MAX_REQUEST_BODY_MIB * 1024 * 1024
_REQUEST_READ_TIMEOUT_SECONDS = 30.0
_SSE_WRITE_TIMEOUT_SECONDS = 30.0


class _ClientConnectionError(Exception):
    pass


class _RequestBodyTooLargeError(ValueError):
    pass


class _RequestReadTimeoutError(TimeoutError):
    pass


class EngineRuntime:
    def __init__(
        self,
        model_kit: object,
        *,
        supports_vision: bool | None = None,
        create_generator_fn: Callable = create_generator,
        get_runtime_load_info_fn: Callable = get_runtime_load_info,
        stop_generation_fn: Callable = stop_generation,
        tokenize_fn: Callable = tokenize,
        unload_fn: Callable = unload,
    ):
        self.model_kit = model_kit
        self.supports_vision = (
            isinstance(model_kit, BatchedVisionModelKit)
            if supports_vision is None
            else supports_vision
        )
        self._create_generator = create_generator_fn
        self._get_runtime_load_info = get_runtime_load_info_fn
        self._stop_generation = stop_generation_fn
        self._tokenize = tokenize_fn
        self._unload = unload_fn

    def prepare_chat_generation(self, body: object) -> ChatGenerationRequest:
        return prepare_chat_generation_request(
            body,
            model_kit=self.model_kit,
            supports_vision=self.supports_vision,
            tokenize=self._tokenize,
        )

    def create_chat_generator(
        self,
        request: ChatGenerationRequest,
        *,
        request_id: str,
        prompt_progress_reporter: PromptProgressReporter,
    ) -> Iterator:
        return self._create_generator(
            self.model_kit,
            request.prompt_tokens,
            request_id=request_id,
            prompt_progress_reporter=prompt_progress_reporter,
            **request.generation_kwargs,
        )

    def runtime_context_length(self) -> int | None:
        return self._get_runtime_load_info(self.model_kit).get("context_length")

    def stop(self, request_id: str) -> None:
        self._stop_generation(self.model_kit, request_id)

    def unload(self) -> None:
        self._unload(self.model_kit)


class GenerationSession:
    def __init__(self, runtime: EngineRuntime):
        self.request_id = str(uuid.uuid4())
        self._runtime = runtime
        self._cancelled = threading.Event()
        self._done = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        if self._cancelled.is_set() or self._done.is_set():
            return
        self._cancelled.set()
        try:
            self._runtime.stop(self.request_id)
        except Exception:
            logger.exception("Failed to cancel MLX generation %s", self.request_id)

    def finish(self) -> None:
        self._done.set()


class _SsePromptProgressReporter(PromptProgressReporter):
    def __init__(
        self,
        handler: MlxEngineRequestHandler,
        session: GenerationSession,
        prompt_tokens: int,
        use_reported_prompt_tokens: bool,
    ):
        self._handler = handler
        self._session = session
        self._use_reported_prompt_tokens = use_reported_prompt_tokens
        self.prompt_tokens = prompt_tokens

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        if not is_draft and self._use_reported_prompt_tokens:
            self.prompt_tokens = total_prompt_tokens
        return self._report(is_draft)

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        return self._report(is_draft)

    def finish(
        self,
        is_draft: bool,
        prefill_tokens_processed: int | None = None,
    ) -> bool:
        return self._report(is_draft)

    def _report(self, is_draft: bool) -> bool:
        if is_draft or self._session.cancelled:
            return not self._session.cancelled
        self._handler._write_bytes(b": prompt-progress\n\n")
        return True


class MlxEngineHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        api_key: str,
        runtime: EngineRuntime,
    ):
        self.api_key = api_key
        self.runtime = runtime
        self._active_sessions: set[GenerationSession] = set()
        self._active_sessions_lock = threading.Lock()
        super().__init__(server_address, MlxEngineRequestHandler)

    def get_request(self):
        connection, client_address = super().get_request()
        connection.settimeout(_REQUEST_READ_TIMEOUT_SECONDS)
        return connection, client_address

    def register_session(self, session: GenerationSession) -> None:
        with self._active_sessions_lock:
            self._active_sessions.add(session)

    def unregister_session(self, session: GenerationSession) -> None:
        with self._active_sessions_lock:
            self._active_sessions.discard(session)

    def cancel_active_sessions(self) -> None:
        with self._active_sessions_lock:
            sessions = list(self._active_sessions)
        for session in sessions:
            session.cancel()


class MlxEngineRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MlxEngineHttpServer

    def do_GET(self) -> None:
        if not self._is_authorized():
            self._send_error(HTTPStatus.UNAUTHORIZED, "Unauthorized.")
            return
        if self.path != "/health":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        body: dict[str, object] = {"status": "ok"}
        context_length = self.server.runtime.runtime_context_length()
        if context_length is not None:
            body["context_length"] = context_length
        self._send_json(HTTPStatus.OK, body)

    def do_POST(self) -> None:
        if not self._is_authorized():
            self._send_error(HTTPStatus.UNAUTHORIZED, "Unauthorized.")
            return
        if self.path != "/v1/chat/completions":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        try:
            request = self.server.runtime.prepare_chat_generation(
                self._read_json_body()
            )
        except _RequestBodyTooLargeError as error:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(error))
            return
        except _RequestReadTimeoutError as error:
            self._send_error(HTTPStatus.REQUEST_TIMEOUT, str(error))
            return
        except (ChatRequestError, ValidationError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        except Exception as error:
            logger.exception("Failed to prepare chat request")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return

        session = GenerationSession(self.server.runtime)
        self.server.register_session(session)
        generator = None
        normal_completion = False
        try:
            self._start_sse_response()
            reporter = _SsePromptProgressReporter(
                self,
                session,
                len(request.prompt_tokens),
                use_reported_prompt_tokens=self.server.runtime.supports_vision,
            )
            generator = self.server.runtime.create_chat_generator(
                request,
                request_id=session.request_id,
                prompt_progress_reporter=reporter,
            )
            self._stream_generation(generator, request, reporter)
            normal_completion = True
        except _ClientConnectionError:
            logger.debug("Generation client disconnected: %s", session.request_id)
        except Exception as error:
            logger.exception("MLX generation failed")
            try:
                self._write_sse_json({"error": {"message": str(error)}})
            except _ClientConnectionError:
                pass
        finally:
            try:
                if not normal_completion:
                    session.cancel()
                if generator is not None:
                    generator.close()
            finally:
                session.finish()
                self.server.unregister_session(session)

    def _stream_generation(
        self,
        generator: Iterator[GenerationResult],
        request: ChatGenerationRequest,
        reporter: _SsePromptProgressReporter,
    ) -> None:
        completion_tokens = 0
        output_parts: list[str] = []
        stop_condition: GenerationStopCondition | None = None
        tool_calling_plan = request.tool_calling_plan

        for result in generator:
            completion_tokens += len(result.tokens)
            if result.text != "":
                if tool_calling_plan.should_buffer_output:
                    output_parts.append(result.text)
                else:
                    self._write_content_delta(result.text)
            if result.stop_condition is not None:
                stop_condition = result.stop_condition

        if stop_condition is None:
            raise RuntimeError("MLX generation ended without a stop condition.")

        if tool_calling_plan.should_buffer_output:
            parsed_tool_calls = tool_calling_plan.parse_output("".join(output_parts))
            if len(parsed_tool_calls.calls) > tool_calling_plan.max_tool_calls:
                raise RuntimeError(
                    "The model produced multiple tool calls, but this server only "
                    "supports one tool call per response."
                )
            if len(parsed_tool_calls.calls) > 0:
                self._write_tool_calls_delta(parsed_tool_calls.calls)
                self._write_sse_json(
                    self._terminal_payload(
                        stop_condition=stop_condition,
                        prompt_tokens=reporter.prompt_tokens,
                        completion_tokens=completion_tokens,
                        finish_reason_override="tool_calls",
                    )
                )
                self._write_bytes(b"data: [DONE]\n\n")
                return

            if parsed_tool_calls.remaining_text != "":
                self._write_content_delta(parsed_tool_calls.remaining_text)

        self._write_sse_json(
            self._terminal_payload(
                stop_condition=stop_condition,
                prompt_tokens=reporter.prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        self._write_bytes(b"data: [DONE]\n\n")

    def _write_content_delta(self, text: str) -> None:
        self._write_sse_json(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ]
            }
        )

    def _write_tool_calls_delta(self, tool_calls: list[dict]) -> None:
        self._write_sse_json(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }
                ]
            }
        )

    def _terminal_payload(
        self,
        *,
        stop_condition: GenerationStopCondition,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason_override: str | None = None,
    ) -> dict:
        stop_reason = stop_condition.stop_reason
        if stop_reason == "eos_token":
            finish_reason = "stop"
            stop_metadata = {"stop_type": "eos"}
        elif stop_reason == "stop_string":
            finish_reason = "stop"
            stop_metadata = {
                "stop_type": "word",
                "stopping_word": stop_condition.stop_string,
            }
        elif stop_reason == "token_limit":
            finish_reason = "length"
            stop_metadata = {"stop_type": "limit"}
        elif stop_reason == "user_cancelled":
            finish_reason = "stop"
            stop_metadata = {"stop_type": "cancel"}
        else:
            raise RuntimeError(f"Unknown MLX stop reason: {stop_reason!r}.")

        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason_override or finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "__lmstudio": stop_metadata,
        }

    def _read_json_body(self) -> object:
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            raise ChatRequestError("Content-Length is required.")
        try:
            content_length = int(content_length_header)
        except ValueError as error:
            raise ChatRequestError(
                "Content-Length must be a positive integer."
            ) from error
        if content_length <= 0:
            raise ChatRequestError("Content-Length must be a positive integer.")
        if content_length > _MAX_REQUEST_BODY_BYTES:
            raise _RequestBodyTooLargeError(
                f"Request body exceeds the {_MAX_REQUEST_BODY_MIB} MiB limit."
            )

        try:
            encoded_body = self.rfile.read(content_length)
        except TimeoutError as error:
            raise _RequestReadTimeoutError("Request body read timed out.") from error
        try:
            return json.loads(encoded_body)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ChatRequestError("The request body must be valid JSON.") from error

    def _is_authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "").encode()
        expected = f"Bearer {self.server.api_key}".encode()
        return hmac.compare_digest(authorization, expected)

    def _start_sse_response(self) -> None:
        try:
            self.connection.settimeout(_SSE_WRITE_TIMEOUT_SECONDS)
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
        except OSError as error:
            raise _ClientConnectionError from error

    def _write_sse_json(self, body: dict) -> None:
        encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self._write_bytes(b"data: " + encoded_body + b"\n\n")

    def _write_bytes(self, content: bytes) -> None:
        try:
            self.wfile.write(content)
            self.wfile.flush()
        except OSError as error:
            raise _ClientConnectionError from error

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(
            status,
            {"error": {"message": message}},
            close_connection=True,
        )

    def _send_json(
        self,
        status: HTTPStatus,
        body: dict,
        *,
        close_connection: bool = False,
    ) -> None:
        encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_body)))
        if close_connection:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(encoded_body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("HTTP %s - %s", self.address_string(), format % args)
