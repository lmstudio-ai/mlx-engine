from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from mlx_engine.generate import (
    create_generator,
    get_runtime_load_info,
    load_model,
    stop_generation,
    tokenize,
    unload,
)
from mlx_engine.openai_tool_calling import (
    ToolCallingPlan,
    ToolCallingValidationError,
    build_tool_calling_plan,
)
from mlx_engine.utils.generation_result import GenerationStopCondition
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]

_FORCED_TOOL_CALL_MISSING_ERROR = "A forced tool_choice was specified, but the model did not produce a valid tool call."


class MlxServerState:
    def __init__(self, *, model_kit: Any, api_key: str | None, model_path: str):
        self.model_kit = model_kit
        self.api_key = api_key
        self.model_path = model_path


class SseWriter:
    def __init__(self, handler: BaseHTTPRequestHandler):
        self._handler = handler
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, payload: JsonObject) -> bool:
        return self._write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n")

    def done(self) -> bool:
        return self._write("data: [DONE]\n\n")

    def _write(self, text: str) -> bool:
        if self._closed:
            return False
        with self._lock:
            if self._closed:
                return False
            try:
                self._handler.wfile.write(text.encode("utf-8"))
                self._handler.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._closed = True
                return False


class SsePromptProgressReporter(PromptProgressReporter):
    def __init__(self, send_progress: Callable[[int, int, int], bool]):
        self._send_progress = send_progress
        self._cached_tokens = 0
        self._total_prompt_tokens = 0

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        if is_draft:
            return True
        self._cached_tokens = max(0, cached_tokens)
        self._total_prompt_tokens = max(0, total_prompt_tokens)
        return self._send_progress(
            self._cached_tokens,
            self._total_prompt_tokens,
            max(0, prefill_tokens_processed),
        )

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        if is_draft:
            return True
        return self._send_progress(
            self._cached_tokens,
            self._total_prompt_tokens,
            max(0, prefill_tokens_processed),
        )

    def finish(
        self, is_draft: bool, prefill_tokens_processed: int | None = None
    ) -> bool:
        if is_draft:
            return True
        processed = (
            self._total_prompt_tokens
            if prefill_tokens_processed is None
            else max(0, prefill_tokens_processed)
        )
        return self._send_progress(
            self._cached_tokens,
            self._total_prompt_tokens,
            processed,
        )


class MlxEngineProtocolHandler(BaseHTTPRequestHandler):
    server_version = "LMStudioMlxEngineProtocol/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    @property
    def state(self) -> MlxServerState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if not self._check_authorization():
            return
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/props":
            runtime_load_info = get_runtime_load_info(self.state.model_kit)
            response = {
                "model_path": self.state.model_path,
                "runtime_load_info": runtime_load_info,
            }
            context_length = runtime_load_info.get("context_length")
            if isinstance(context_length, int):
                response["default_generation_settings"] = {"n_ctx": context_length}
            self._send_json(response)
            return
        self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}")

    def do_POST(self) -> None:
        if not self._check_authorization():
            return
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()
            if path == "/tokenize":
                self._handle_tokenize(body)
                return
            if path == "/apply-template":
                self._handle_apply_template(body)
                return
            if path == "/v1/chat/completions":
                self._handle_chat_completions(body)
                return
            if path == "/v1/completions":
                self._handle_text_completions(body)
                return
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}")
        except HttpRequestError as error:
            self._send_error(error.status, error.message)
        except Exception as error:
            logger.exception("Request failed")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def _check_authorization(self) -> bool:
        api_key = self.state.api_key
        if api_key is None or api_key == "":
            return True
        expected = f"Bearer {api_key}"
        supplied = self.headers.get("Authorization", "")
        if supplied == expected:
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("content-type", "application/json")
        self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Unauthorized"}).encode("utf-8"))
        return False

    def _read_json_body(self) -> JsonObject:
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            return {}
        try:
            content_length = int(content_length_header)
        except ValueError as error:
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "Invalid Content-Length"
            ) from error
        raw_body = self.rfile.read(content_length)
        if len(raw_body) == 0:
            return {}
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "Malformed JSON request body"
            ) from error
        if not isinstance(body, dict):
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "JSON request body must be an object"
            )
        return body

    def _handle_tokenize(self, body: JsonObject) -> None:
        content = body.get("content")
        if not isinstance(content, str):
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "tokenize.content must be a string"
            )
        tokens = tokenize(self.state.model_kit, content)
        self._send_json({"tokens": tokens})

    def _handle_apply_template(self, body: JsonObject) -> None:
        messages = get_required_list(body, "messages")
        tools = get_optional_list(body, "tools")
        tool_calling_plan = get_tool_calling_plan(
            messages=messages,
            tools=tools,
            body=body,
            response_json_schema=None,
        )
        chat_template_kwargs = get_optional_dict(body, "chat_template_kwargs") or {}
        rendered_prompt, _images_b64 = render_chat_prompt(
            self.state.model_kit,
            messages=tool_calling_plan.prompt_messages,
            tools=tool_calling_plan.template_tools,
            tool_choice=tool_calling_plan.template_tool_choice,
            add_generation_prompt=not messages_end_with_assistant(messages),
            continue_final_message=messages_end_with_assistant(messages),
            chat_template_kwargs=chat_template_kwargs,
        )
        self._send_json({"prompt": rendered_prompt})

    def _handle_chat_completions(self, body: JsonObject) -> None:
        if body.get("stream") is not True:
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST,
                "mlx-server only supports streaming chat completions",
            )
        request_id = str(uuid.uuid4())
        messages = get_required_list(body, "messages")
        chat_template_kwargs = get_optional_dict(body, "chat_template_kwargs") or {}
        tools = get_optional_list(body, "tools")
        tool_calling_plan = get_tool_calling_plan(
            messages=messages,
            tools=tools,
            body=body,
            response_json_schema=get_json_schema(body),
        )
        prompt, images_b64 = render_chat_prompt(
            self.state.model_kit,
            messages=tool_calling_plan.prompt_messages,
            tools=tool_calling_plan.template_tools,
            tool_choice=tool_calling_plan.template_tool_choice,
            add_generation_prompt=get_bool(body, "add_generation_prompt", True),
            continue_final_message=get_bool(body, "continue_final_message", False),
            chat_template_kwargs=chat_template_kwargs,
        )
        prompt_tokens = tokenize(self.state.model_kit, prompt)
        self._start_sse_response()
        writer = SseWriter(self)
        self._stream_generation(
            writer=writer,
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            body=body,
            tool_calling_plan=tool_calling_plan,
            event_builder=ChatCompletionEventBuilder(model=body.get("model")),
        )

    def _handle_text_completions(self, body: JsonObject) -> None:
        if body.get("stream") is not True:
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST,
                "mlx-server only supports streaming text completions",
            )
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "completions.prompt must be a string"
            )
        request_id = str(uuid.uuid4())
        prompt_tokens = tokenize(self.state.model_kit, prompt)
        self._start_sse_response()
        writer = SseWriter(self)
        self._stream_generation(
            writer=writer,
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            images_b64=[],
            body=body,
            tool_calling_plan=ToolCallingPlan(
                strategy="none",
                tool_specs=[],
                prompt_messages=[],
                template_tools=None,
                template_tool_choice=None,
                generation_json_schema=get_json_schema(body),
            ),
            event_builder=TextCompletionEventBuilder(model=body.get("model")),
        )

    def _stream_generation(
        self,
        *,
        writer: SseWriter,
        request_id: str,
        prompt: str,
        prompt_tokens: list[int],
        images_b64: list[str],
        body: JsonObject,
        tool_calling_plan: ToolCallingPlan,
        event_builder: CompletionEventBuilder,
    ) -> None:
        prompt_tokens_count = len(prompt_tokens)
        predicted_tokens_count = 0
        raw_output_text_parts: list[str] = []
        visible_output_text_parts: list[str] = []
        should_buffer_output = tool_calling_plan.should_buffer_output
        stop_condition: GenerationStopCondition | None = None
        started_at = time.monotonic()
        first_token_at: float | None = None

        def send_progress(cached: int, total: int, processed: int) -> bool:
            return writer.send(
                event_builder.progress_event(
                    cached_prompt_tokens_count=cached,
                    total_prompt_tokens_count=total,
                    processed_prompt_tokens_count=processed,
                )
            )

        writer.send(event_builder.model_input_event(prompt))
        generator: Iterable[Any] | None = None
        try:
            generator = create_generator(
                self.state.model_kit,
                prompt_tokens,
                images_b64=images_b64,
                stop_strings=get_stop_strings(body),
                top_logprobs=get_optional_int(body, "top_logprobs"),
                repetition_penalty=get_optional_number(body, "repetition_penalty"),
                temp=get_optional_number(body, "temperature"),
                top_p=get_optional_number(body, "top_p"),
                top_k=get_optional_int(body, "top_k"),
                min_p=get_optional_number(body, "min_p"),
                seed=get_optional_int(body, "seed"),
                json_schema=tool_calling_plan.generation_json_schema,
                max_tokens=get_max_tokens(body),
                request_id=request_id,
                prompt_progress_reporter=SsePromptProgressReporter(send_progress),
            )
            for generation_result in generator:
                if writer.closed:
                    stop_generation(self.state.model_kit, request_id)
                    break
                if generation_result.text != "":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    raw_output_text_parts.append(generation_result.text)
                    predicted_tokens_count += len(generation_result.tokens)
                    if not should_buffer_output:
                        visible_output_text_parts.append(generation_result.text)
                        writer.send(
                            event_builder.content_event(
                                text=generation_result.text,
                                tokens=generation_result.tokens,
                                top_logprobs=generation_result.top_logprobs,
                            )
                        )
                if generation_result.stop_condition is not None:
                    stop_condition = generation_result.stop_condition
            if writer.closed:
                return
            raw_output = "".join(raw_output_text_parts)
            parsed_tool_calls = tool_calling_plan.parse_output(raw_output)
            completed_at = time.monotonic()
            prompt_time_ms = (
                completed_at - started_at
                if first_token_at is None
                else first_token_at - started_at
            ) * 1000
            predicted_time_ms = (
                0 if first_token_at is None else (completed_at - first_token_at) * 1000
            )

            if len(parsed_tool_calls.calls) > 0:
                writer.send(
                    event_builder.tool_calls_delta_event(
                        tool_calls=parsed_tool_calls.calls,
                    )
                )
                writer.send(
                    event_builder.tool_calls_terminal_event(
                        prompt=prompt,
                        output=parsed_tool_calls.remaining_text,
                        prompt_tokens_count=prompt_tokens_count,
                        predicted_tokens_count=predicted_tokens_count,
                        prompt_time_ms=prompt_time_ms,
                        predicted_time_ms=predicted_time_ms,
                        stop_condition=stop_condition,
                    )
                )
                writer.done()
                return

            if tool_calling_plan.requires_tool_call:
                raise RuntimeError(_FORCED_TOOL_CALL_MISSING_ERROR)

            if should_buffer_output and parsed_tool_calls.remaining_text != "":
                visible_output_text_parts.append(parsed_tool_calls.remaining_text)
                writer.send(
                    event_builder.content_event(
                        text=parsed_tool_calls.remaining_text,
                        tokens=[],
                        top_logprobs=[],
                    )
                )

            writer.send(
                event_builder.terminal_event(
                    prompt=prompt,
                    output="".join(visible_output_text_parts),
                    prompt_tokens_count=prompt_tokens_count,
                    predicted_tokens_count=predicted_tokens_count,
                    prompt_time_ms=prompt_time_ms,
                    predicted_time_ms=predicted_time_ms,
                    stop_condition=stop_condition,
                )
            )
            writer.done()
        except Exception as error:
            logger.exception("Streaming generation failed")
            if not writer.closed:
                writer.send({"error": {"message": str(error)}})
                writer.done()
        finally:
            if generator is not None:
                close_generator(generator)

    def _start_sse_response(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers()

    def _send_json(
        self, payload: JsonObject, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        response_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": {"message": message}}, status=status)


class CompletionEventBuilder:
    def __init__(self, *, model: Any, object_name: str):
        self._model = model if isinstance(model, str) else "mlx"
        self._object_name = object_name
        self._created = int(time.time())
        self._completion_id = f"cmpl-{uuid.uuid4().hex}"

    def progress_event(
        self,
        *,
        cached_prompt_tokens_count: int,
        total_prompt_tokens_count: int,
        processed_prompt_tokens_count: int,
    ) -> JsonObject:
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "choices": [],
            "prompt_progress": {
                "cache": cached_prompt_tokens_count,
                "total": total_prompt_tokens_count,
                "processed": processed_prompt_tokens_count,
            },
        }

    def model_input_event(self, prompt: str) -> JsonObject:
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "choices": [],
            "__lmstudio": {"model_input": prompt},
        }

    def _base_terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        total_tokens_count = prompt_tokens_count + predicted_tokens_count
        predicted_seconds = predicted_time_ms / 1000
        predicted_per_second = (
            predicted_tokens_count / predicted_seconds if predicted_seconds > 0 else 0
        )
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "usage": {
                "prompt_tokens": prompt_tokens_count,
                "completion_tokens": predicted_tokens_count,
                "total_tokens": total_tokens_count,
            },
            "timings": {
                "prompt_ms": prompt_time_ms,
                "predicted_ms": predicted_time_ms,
                "predicted_per_second": predicted_per_second,
            },
            "__lmstudio": {
                "model_input": prompt,
                "model_output": output,
                **to_lmstudio_stop_metadata(stop_condition),
            },
        }

    def content_event(
        self,
        *,
        text: str,
        tokens: list[Any],
        top_logprobs: list[list[Any]],
    ) -> JsonObject:
        raise NotImplementedError

    def terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        raise NotImplementedError

    def tool_calls_delta_event(
        self,
        *,
        tool_calls: list[JsonObject],
    ) -> JsonObject:
        raise NotImplementedError

    def tool_calls_terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        raise NotImplementedError


class ChatCompletionEventBuilder(CompletionEventBuilder):
    def __init__(self, *, model: Any):
        super().__init__(model=model, object_name="chat.completion.chunk")

    def content_event(
        self,
        *,
        text: str,
        tokens: list[Any],
        top_logprobs: list[list[Any]],
    ) -> JsonObject:
        choice: JsonObject = {
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }
        logprobs = to_openai_logprobs(tokens, top_logprobs)
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "choices": [choice],
        }

    def terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        event = self._base_terminal_event(
            prompt=prompt,
            output=output,
            prompt_tokens_count=prompt_tokens_count,
            predicted_tokens_count=predicted_tokens_count,
            prompt_time_ms=prompt_time_ms,
            predicted_time_ms=predicted_time_ms,
            stop_condition=stop_condition,
        )
        event["choices"] = [
            {
                "index": 0,
                "delta": {},
                "finish_reason": to_openai_finish_reason(stop_condition),
            }
        ]
        return event

    def tool_calls_delta_event(
        self,
        *,
        tool_calls: list[JsonObject],
    ) -> JsonObject:
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "tool_calls": tool_calls},
                    "finish_reason": None,
                }
            ],
        }

    def tool_calls_terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        event = self._base_terminal_event(
            prompt=prompt,
            output=output,
            prompt_tokens_count=prompt_tokens_count,
            predicted_tokens_count=predicted_tokens_count,
            prompt_time_ms=prompt_time_ms,
            predicted_time_ms=predicted_time_ms,
            stop_condition=stop_condition,
        )
        event["choices"] = [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ]
        return event


class TextCompletionEventBuilder(CompletionEventBuilder):
    def __init__(self, *, model: Any):
        super().__init__(model=model, object_name="text_completion")

    def content_event(
        self,
        *,
        text: str,
        tokens: list[Any],
        top_logprobs: list[list[Any]],
    ) -> JsonObject:
        choice: JsonObject = {
            "index": 0,
            "text": text,
            "finish_reason": None,
        }
        logprobs = to_openai_logprobs(tokens, top_logprobs)
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
        return {
            "id": self._completion_id,
            "object": self._object_name,
            "created": self._created,
            "model": self._model,
            "choices": [choice],
        }

    def terminal_event(
        self,
        *,
        prompt: str,
        output: str,
        prompt_tokens_count: int,
        predicted_tokens_count: int,
        prompt_time_ms: float,
        predicted_time_ms: float,
        stop_condition: GenerationStopCondition | None,
    ) -> JsonObject:
        event = self._base_terminal_event(
            prompt=prompt,
            output=output,
            prompt_tokens_count=prompt_tokens_count,
            predicted_tokens_count=predicted_tokens_count,
            prompt_time_ms=prompt_time_ms,
            predicted_time_ms=predicted_time_ms,
            stop_condition=stop_condition,
        )
        event["choices"] = [
            {
                "index": 0,
                "text": "",
                "finish_reason": to_openai_finish_reason(stop_condition),
            }
        ]
        return event


class HttpRequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class EngineProtocolHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: MlxServerState):
        super().__init__(server_address, MlxEngineProtocolHandler)
        self.state = state


def get_tool_calling_plan(
    *,
    messages: list[Any],
    tools: list[Any] | None,
    body: JsonObject,
    response_json_schema: str | None,
) -> ToolCallingPlan:
    try:
        return build_tool_calling_plan(
            messages=messages,
            tools=tools,
            tool_choice_value=body.get("tool_choice"),
            parallel_tool_calls=get_parallel_tool_calls(body),
            response_json_schema=response_json_schema,
        )
    except ToolCallingValidationError as error:
        raise HttpRequestError(HTTPStatus.BAD_REQUEST, str(error)) from error


def get_parallel_tool_calls(body: JsonObject) -> bool:
    return get_bool(body, "parallel_tool_calls", True)


def chat_template_type_error(
    error: TypeError,
    tools: list[Any] | None,
    tool_choice: Any,
) -> HttpRequestError:
    if (tools is not None and len(tools) > 0) or tool_choice is not None:
        return HttpRequestError(
            HTTPStatus.BAD_REQUEST,
            f"Chat template rejected tool parameters: {error}",
        )
    return HttpRequestError(
        HTTPStatus.BAD_REQUEST,
        f"Chat template rejected request parameters: {error}",
    )


def render_chat_prompt(
    model_kit: Any,
    *,
    messages: list[Any],
    tools: list[Any] | None,
    tool_choice: Any,
    add_generation_prompt: bool,
    continue_final_message: bool,
    chat_template_kwargs: JsonObject,
) -> tuple[str, list[str]]:
    normalized_messages, images_b64 = normalize_messages_for_chat_template(messages)
    template_owner = get_chat_template_owner(model_kit)
    template_kwargs: JsonObject = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        **chat_template_kwargs,
    }
    if continue_final_message:
        template_kwargs["continue_final_message"] = True
    if tools is not None and len(tools) > 0:
        template_kwargs["tools"] = tools
    if tool_choice is not None:
        template_kwargs["tool_choice"] = tool_choice

    try:
        prompt = template_owner.apply_chat_template(
            normalized_messages, **template_kwargs
        )
    except TypeError as first_error:
        if not continue_final_message:
            raise chat_template_type_error(
                first_error, tools, tool_choice
            ) from first_error
        template_kwargs.pop("continue_final_message", None)
        try:
            prompt = template_owner.apply_chat_template(
                normalized_messages, **template_kwargs
            )
        except TypeError as second_error:
            raise chat_template_type_error(
                second_error, tools, tool_choice
            ) from second_error
    if not isinstance(prompt, str):
        raise HttpRequestError(
            HTTPStatus.BAD_REQUEST, "Chat template did not return a string"
        )
    return prompt, images_b64


def get_chat_template_owner(model_kit: Any) -> Any:
    processor = getattr(model_kit, "processor", None)
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor
    tokenizer = getattr(model_kit, "tokenizer", None)
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if raw_tokenizer is not None and hasattr(raw_tokenizer, "apply_chat_template"):
        return raw_tokenizer
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise HttpRequestError(
        HTTPStatus.BAD_REQUEST, "Loaded tokenizer has no chat template"
    )


def normalize_messages_for_chat_template(
    messages: list[Any],
) -> tuple[list[JsonObject], list[str]]:
    normalized_messages: list[JsonObject] = []
    images_b64: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "messages entries must be objects"
            )
        role = message.get("role")
        if not isinstance(role, str):
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST, "message.role must be a string"
            )
        normalized_message = {
            key: value for key, value in message.items() if key != "content"
        }
        normalized_message["content"] = normalize_message_content(
            message.get("content"), images_b64
        )
        if "tool_calls" in normalized_message:
            normalized_message["tool_calls"] = normalize_tool_calls(
                normalized_message["tool_calls"]
            )
        normalized_messages.append(normalized_message)
    return normalized_messages, images_b64


def normalize_message_content(content: Any, images_b64: list[str]) -> Any:
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        return content

    normalized_parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            normalized_parts.append(part)
            continue
        part_type = part.get("type")
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if not isinstance(url, str):
                raise HttpRequestError(
                    HTTPStatus.BAD_REQUEST, "image_url.url must be a string"
                )
            image_b64 = extract_base64_image_payload(url)
            images_b64.append(image_b64)
            normalized_parts.append({"type": "image", "base64": image_b64})
        elif part_type == "image" and isinstance(part.get("base64"), str):
            image_b64 = part["base64"]
            images_b64.append(image_b64)
            normalized_parts.append(part)
        else:
            normalized_parts.append(part)
    return normalized_parts


def normalize_tool_calls(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls

    normalized_tool_calls: list[Any] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
            continue
        normalized_tool_call = dict(tool_call)
        function = normalized_tool_call.get("function")
        if isinstance(function, dict):
            normalized_function = dict(function)
            arguments = normalized_function.get("arguments")
            if isinstance(arguments, str):
                try:
                    normalized_function["arguments"] = (
                        {} if arguments == "" else json.loads(arguments)
                    )
                except json.JSONDecodeError as error:
                    raise HttpRequestError(
                        HTTPStatus.BAD_REQUEST,
                        "tool_calls.function.arguments must be valid JSON",
                    ) from error
            normalized_tool_call["function"] = normalized_function
        normalized_tool_calls.append(normalized_tool_call)
    return normalized_tool_calls


def extract_base64_image_payload(url: str) -> str:
    if url.startswith("data:"):
        marker = ";base64,"
        marker_index = url.find(marker)
        if marker_index < 0:
            raise HttpRequestError(
                HTTPStatus.BAD_REQUEST,
                "image_url data URI must contain a base64 payload",
            )
        return url[marker_index + len(marker) :]
    try:
        base64.b64decode(url, validate=True)
    except Exception as error:
        raise HttpRequestError(
            HTTPStatus.BAD_REQUEST,
            "image_url must be an inline base64 data URI for mlx-server",
        ) from error
    return url


def messages_end_with_assistant(messages: list[Any]) -> bool:
    if len(messages) == 0:
        return False
    last_message = messages[-1]
    return isinstance(last_message, dict) and last_message.get("role") == "assistant"


def get_stop_strings(body: JsonObject) -> list[str] | None:
    stop = body.get("stop")
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(entry, str) for entry in stop):
        return stop
    raise HttpRequestError(
        HTTPStatus.BAD_REQUEST, "stop must be a string or string array"
    )


def get_json_schema(body: JsonObject) -> str | None:
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    response_format_type = response_format.get("type")
    if response_format_type == "json_object":
        return json.dumps({})
    if response_format_type != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict) and "schema" in json_schema:
        return json.dumps(json_schema["schema"])
    if isinstance(json_schema, dict):
        return json.dumps(json_schema)
    raise HttpRequestError(
        HTTPStatus.BAD_REQUEST, "response_format.json_schema must be an object"
    )


def get_max_tokens(body: JsonObject) -> int:
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        return 10_000_000
    if isinstance(max_tokens, int) and max_tokens >= 0:
        return max_tokens
    raise HttpRequestError(
        HTTPStatus.BAD_REQUEST, "max_tokens must be a non-negative integer"
    )


def get_required_list(body: JsonObject, field_name: str) -> list[Any]:
    value = body.get(field_name)
    if not isinstance(value, list):
        raise HttpRequestError(HTTPStatus.BAD_REQUEST, f"{field_name} must be an array")
    return value


def get_optional_list(body: JsonObject, field_name: str) -> list[Any] | None:
    value = body.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise HttpRequestError(HTTPStatus.BAD_REQUEST, f"{field_name} must be an array")
    return value


def get_optional_dict(body: JsonObject, field_name: str) -> JsonObject | None:
    value = body.get(field_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HttpRequestError(
            HTTPStatus.BAD_REQUEST, f"{field_name} must be an object"
        )
    return value


def get_bool(body: JsonObject, field_name: str, default: bool) -> bool:
    value = body.get(field_name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise HttpRequestError(
            HTTPStatus.BAD_REQUEST, f"{field_name} must be a boolean"
        )
    return value


def get_optional_number(body: JsonObject, field_name: str) -> float | None:
    value = body.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HttpRequestError(HTTPStatus.BAD_REQUEST, f"{field_name} must be a number")
    return float(value)


def get_optional_int(body: JsonObject, field_name: str) -> int | None:
    value = body.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise HttpRequestError(
            HTTPStatus.BAD_REQUEST, f"{field_name} must be an integer"
        )
    return value


def to_openai_finish_reason(stop_condition: GenerationStopCondition | None) -> str:
    if stop_condition is None:
        return "stop"
    if stop_condition.stop_reason == "token_limit":
        return "length"
    return "stop"


def to_lmstudio_stop_metadata(
    stop_condition: GenerationStopCondition | None,
) -> JsonObject:
    if stop_condition is None:
        return {"stop_type": "eos"}
    if stop_condition.stop_reason == "stop_string":
        return {
            "stop_type": "word",
            "stopping_word": stop_condition.stop_string,
        }
    if stop_condition.stop_reason == "token_limit":
        return {"stop_type": "limit"}
    if stop_condition.stop_reason == "user_cancelled":
        return {"stop_type": "user_cancelled"}
    return {"stop_type": "eos"}


def to_openai_logprobs(
    tokens: list[Any], top_logprobs: list[list[Any]]
) -> list[JsonObject] | None:
    if len(tokens) == 0:
        return None
    logprobs: list[JsonObject] = []
    for token_index, token in enumerate(tokens):
        token_data = token_to_json(token)
        if token_index < len(top_logprobs):
            token_data["top_logprobs"] = [
                token_to_json(candidate) for candidate in top_logprobs[token_index]
            ]
        logprobs.append(token_data)
    return logprobs


def token_to_json(token: Any) -> JsonObject:
    if hasattr(token, "__dataclass_fields__"):
        token_dict = asdict(token)
    elif isinstance(token, dict):
        token_dict = token
    else:
        token_dict = {}
    return {
        "id": token_dict.get("id", 0),
        "token": token_dict.get("text", ""),
        "logprob": token_dict.get("logprob", 0),
    }


def close_generator(generator: Iterable[Any]) -> None:
    close = getattr(generator, "close", None)
    if callable(close):
        close()


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LM Studio MLX engine protocol server")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", required=True, type=int, help="HTTP bind port")
    parser.add_argument(
        "--api-key", default=None, help="Bearer token required for requests"
    )
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--max-seq-nums", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=None)
    parser.add_argument("--quantized-kv-start", type=int, default=None)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    return parser


def configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(level=level)


def main() -> int:
    parser = create_argument_parser()
    args = parser.parse_args()
    configure_logging(args.verbosity)
    logger.info("Loading MLX model from %s", args.model)
    model_kit = load_model(
        args.model,
        max_kv_size=args.max_kv_size,
        max_seq_nums=args.max_seq_nums,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
    )
    state = MlxServerState(
        model_kit=model_kit, api_key=args.api_key, model_path=args.model
    )
    http_server = EngineProtocolHttpServer((args.host, args.port), state)
    logger.info("MLX engine protocol server listening on %s:%s", args.host, args.port)
    try:
        http_server.serve_forever()
    finally:
        unload(model_kit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
