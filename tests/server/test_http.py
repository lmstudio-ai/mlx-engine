from contextlib import contextmanager
import http.client
import json
import socket
import struct
import threading
import time
import weakref

import mlx_engine.server.http as server_http
import pytest
from mlx_engine.server.http import (
    EngineRuntime,
    GenerationSession,
    MlxEngineHttpServer,
)
from mlx_engine.utils.generation_result import (
    GenerationResult,
    GenerationStopCondition,
)
from mlx_engine.utils.prompt_progress_reporter import BatchedMlxLmReporterAdapter
from mlx_engine.utils.token import Token


_DECOMPRESSION_BOMB_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAJxAAACcQCAIAAAA1LPVwAAAAAElFTkSuQmCC"
)
_TRUNCATED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQ="


class _FakeRenderer:
    chat_template = "model template"

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Hello"}]
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        return "rendered prompt"


class _FakeTokenizer:
    def __init__(self):
        self._tokenizer = _FakeRenderer()


class _FakeModelKit:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeVisionModelKit:
    def __init__(self):
        self.processor = _FakeRenderer()


def _request_body():
    return {
        "model": "single-loaded-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6,
        "max_tokens": 32,
        "stop": ["END"],
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.03,
        "repeat_penalty": 1.05,
    }


def _parse_sse(response_text):
    events = []
    for block in response_text.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data != "[DONE]":
                events.append(json.loads(data))
    return events


@contextmanager
def _running_server(runtime, *, send_buffer_size=None):
    server = MlxEngineHttpServer(
        ("127.0.0.1", 0),
        api_key="secret-token",
        runtime=runtime,
    )
    if send_buffer_size is not None:
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer_size)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.cancel_active_sessions()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(port, method, path, *, body=None, authorized=True):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {}
    if authorized:
        headers["Authorization"] = "Bearer secret-token"
    if body is not None:
        headers["Content-Type"] = "application/json"
        encoded_body = json.dumps(body)
    else:
        encoded_body = None
    connection.request(method, path, body=encoded_body, headers=headers)
    response = connection.getresponse()
    response_body = response.read().decode("utf-8")
    connection.close()
    return response.status, response_body


def _request_with_content_length(port, content_length):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer secret-token",
            "Content-Type": "application/json",
            "Content-Length": content_length,
        },
    )
    response = connection.getresponse()
    response_body = response.read().decode("utf-8")
    connection.close()
    return response.status, response_body


def test_health_requires_auth_and_reports_actualized_context_length():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {"context_length": 8192},
    )

    with _running_server(runtime) as port:
        status, body = _request(port, "GET", "/health", authorized=False)
        assert status == 401
        assert json.loads(body) == {"error": {"message": "Unauthorized."}}

        status, body = _request(port, "GET", "/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok", "context_length": 8192}


def test_non_ascii_authorization_is_rejected():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )

    with _running_server(runtime) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(
            "GET",
            "/health",
            headers={"Authorization": "Bearer \xff"},
        )
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        connection.close()

    assert response.status == 401
    assert json.loads(response_body) == {"error": {"message": "Unauthorized."}}


def test_partial_headers_time_out(monkeypatch):
    monkeypatch.setattr(server_http, "_REQUEST_READ_TIMEOUT_SECONDS", 0.05)
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )

    with _running_server(runtime) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(b"GET /health HTTP/1.1\r\n")
            assert client.recv(1) == b""


def test_partial_request_body_returns_request_timeout(monkeypatch):
    monkeypatch.setattr(server_http, "_REQUEST_READ_TIMEOUT_SECONDS", 0.05)
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )

    with _running_server(runtime) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer secret-token\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 2\r\n\r\n"
                b"{"
            )
            response = b""
            while chunk := client.recv(4096):
                response += chunk

    assert b"HTTP/1.1 408 Request Timeout" in response
    assert b"Request body read timed out." in response


def test_rejected_post_closes_connection_before_unread_body_can_be_reused():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )
    body = b'{"messages":[]}'

    with _running_server(runtime) as port:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                + body
                + b"GET /health HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer secret-token\r\n\r\n"
            )
            response = b""
            while chunk := client.recv(4096):
                response += chunk

    assert response.count(b"HTTP/1.1") == 1
    assert b"HTTP/1.1 401 Unauthorized" in response
    assert b"\r\nConnection: close\r\n" in response


def test_invalid_and_oversized_content_lengths_are_rejected():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )

    with _running_server(runtime) as port:
        for content_length in ("invalid", "-1", "0"):
            status, body = _request_with_content_length(port, content_length)
            assert status == 400
            assert json.loads(body) == {
                "error": {"message": "Content-Length must be a positive integer."}
            }

        status, body = _request_with_content_length(
            port,
            str(server_http._MAX_REQUEST_BODY_BYTES + 1),
        )
        assert status == 413
        assert json.loads(body) == {
            "error": {
                "message": (
                    "Request body exceeds the "
                    f"{server_http._MAX_REQUEST_BODY_MIB} MiB limit."
                )
            }
        }


def test_parsed_request_body_is_released_before_generation(monkeypatch):
    class WeakReferenceableDict(dict):
        pass

    pending_bodies = [WeakReferenceableDict(_request_body())]
    body_reference = weakref.ref(pending_bodies[0])
    body_released_before_generation = []
    original_json_loads = json.loads

    def parse_request_body(value):
        if isinstance(value, bytes):
            return pending_bodies.pop()
        return original_json_loads(value)

    monkeypatch.setattr(server_http.json, "loads", parse_request_body)

    def create_generator(_model_kit, _prompt_tokens, **_kwargs):
        body_released_before_generation.append(body_reference() is None)
        yield GenerationResult(
            text="",
            tokens=[],
            top_logprobs=[],
            stop_condition=GenerationStopCondition(
                stop_reason="eos_token",
                stop_string="",
                stop_tokens=[2],
            ),
        )

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        tokenize_fn=lambda _model_kit, _prompt: [1],
    )

    with _running_server(runtime) as port:
        status, _response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=_request_body(),
        )

    assert status == 200
    assert body_released_before_generation == [True]


def test_invalid_generation_settings_are_rejected_before_streaming():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )

    with _running_server(runtime) as port:
        invalid_body = _request_body()
        invalid_body["temperature"] = -0.1
        status, response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=invalid_body,
        )
        assert status == 400
        assert "temperature" in json.loads(response_body)["error"]["message"]

        empty_stop_body = _request_body()
        empty_stop_body["stop"] = [""]
        status, response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=empty_stop_body,
        )
        assert status == 400
        assert "stop" in json.loads(response_body)["error"]["message"]

        unsupported_body = _request_body()
        unsupported_body["max_completion_tokens"] = 1
        status, response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=unsupported_body,
        )
        assert status == 400
        assert json.loads(response_body) == {
            "error": {
                "message": "Unsupported generation controls: max_completion_tokens."
            }
        }


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(
    ("image_data", "error_message"),
    [
        ("not-valid-base64!", "Images must contain valid base64 data."),
        ("bm90IGFuIGltYWdl", "Images must contain supported image data."),
        (_TRUNCATED_PNG_B64, "Images must contain supported image data."),
        (_DECOMPRESSION_BOMB_PNG_B64, "Image dimensions are too large."),
    ],
)
def test_invalid_image_is_rejected_before_streaming(image_data, error_message):
    runtime = EngineRuntime(
        _FakeVisionModelKit(),
        supports_vision=True,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )
    body = _request_body()
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            ],
        }
    ]

    with _running_server(runtime) as port:
        status, response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=body,
        )

    assert status == 400
    assert json.loads(response_body) == {"error": {"message": error_message}}


def test_chat_stream_forwards_generation_settings_and_returns_usage():
    generation_calls = []

    def create_generator(model_kit, prompt_tokens, **kwargs):
        generation_calls.append((model_kit, prompt_tokens, kwargs))
        reporter = kwargs["prompt_progress_reporter"]
        assert reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=3,
            prefill_tokens_processed=0,
        )
        assert reporter.update(is_draft=False, prefill_tokens_processed=2)
        yield GenerationResult(
            text="Hello back",
            tokens=[
                Token(id=10, text="Hello", logprob=-0.1),
                Token(id=11, text=" back", logprob=-0.2),
            ],
            top_logprobs=[],
            stop_condition=None,
        )
        yield GenerationResult(
            text="",
            tokens=[],
            top_logprobs=[],
            stop_condition=GenerationStopCondition(
                stop_reason="eos_token",
                stop_string="",
                stop_tokens=[2],
            ),
        )

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        tokenize_fn=lambda _model_kit, prompt: [1, 2, 3]
        if prompt == "rendered prompt"
        else [],
    )

    with _running_server(runtime) as port:
        status, response_text = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body={
                **_request_body(),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                        }
                    },
                },
            },
        )

    assert status == 200
    assert ": prompt-progress\n\n" in response_text
    assert response_text.endswith("data: [DONE]\n\n")
    events = _parse_sse(response_text)
    assert events[0] == {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello back"},
                "finish_reason": None,
            }
        ]
    }
    assert events[1] == {
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        "__lmstudio": {"stop_type": "eos"},
    }

    model_kit, prompt_tokens, generation_kwargs = generation_calls[0]
    assert isinstance(model_kit, _FakeModelKit)
    assert prompt_tokens == [1, 2, 3]
    assert generation_kwargs["request_id"] != ""
    assert generation_kwargs["images_b64"] == []
    assert generation_kwargs["temp"] == 0.6
    assert generation_kwargs["max_tokens"] == 32
    assert generation_kwargs["stop_strings"] == ["END"]
    assert generation_kwargs["top_p"] == 0.9
    assert generation_kwargs["top_k"] == 20
    assert generation_kwargs["min_p"] == 0.03
    assert generation_kwargs["repetition_penalty"] == 1.05
    assert json.loads(generation_kwargs["json_schema"]) == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


def test_batched_text_cache_hit_preserves_full_prompt_usage():
    request_count = 0

    def create_generator(_model_kit, prompt_tokens, **kwargs):
        nonlocal request_count
        reporter = BatchedMlxLmReporterAdapter(
            kwargs["prompt_progress_reporter"],
            emit_begin=True,
        )
        if request_count == 0:
            assert reporter(0, len(prompt_tokens))
            assert reporter(len(prompt_tokens) - 1, len(prompt_tokens))
        else:
            assert reporter(1, 1)
        request_count += 1
        yield GenerationResult(
            text="",
            tokens=[],
            top_logprobs=[],
            stop_condition=GenerationStopCondition(
                stop_reason="eos_token",
                stop_string="",
                stop_tokens=[2],
            ),
        )

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        tokenize_fn=lambda _model_kit, _prompt: [1, 2, 3],
    )

    with _running_server(runtime) as port:
        responses = [
            _request(
                port,
                "POST",
                "/v1/chat/completions",
                body=_request_body(),
            )
            for _ in range(2)
        ]

    assert request_count == 2
    for status, response_text in responses:
        assert status == 200
        terminal_event = _parse_sse(response_text)[0]
        assert terminal_event["usage"]["prompt_tokens"] == 3


def test_vision_usage_uses_the_prepared_prompt_length():
    def create_generator(_model_kit, _prompt_tokens, **kwargs):
        reporter = kwargs["prompt_progress_reporter"]
        assert reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=9,
            prefill_tokens_processed=0,
        )
        yield GenerationResult(
            text="",
            tokens=[],
            top_logprobs=[],
            stop_condition=GenerationStopCondition(
                stop_reason="eos_token",
                stop_string="",
                stop_tokens=[2],
            ),
        )

    runtime = EngineRuntime(
        _FakeVisionModelKit(),
        supports_vision=True,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        tokenize_fn=lambda _model_kit, _prompt: [1, 2, 3],
    )

    with _running_server(runtime) as port:
        status, response_text = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=_request_body(),
        )

    assert status == 200
    terminal_event = _parse_sse(response_text)[0]
    assert terminal_event["usage"]["prompt_tokens"] == 9


def test_tools_are_rejected_before_streaming():
    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        get_runtime_load_info_fn=lambda _model_kit: {},
    )
    body = _request_body()
    body["tools"] = [{"type": "function", "function": {"name": "search"}}]

    with _running_server(runtime) as port:
        status, response_body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=body,
        )

    assert status == 400
    assert json.loads(response_body) == {
        "error": {"message": "Tools are not supported yet."}
    }


def test_generation_errors_are_returned_inside_the_stream():
    for generation_error in (
        RuntimeError("generation failed"),
        OSError("backend I/O failed"),
    ):

        def create_generator(_model_kit, _prompt_tokens, **_kwargs):
            raise generation_error
            yield

        runtime = EngineRuntime(
            _FakeModelKit(),
            supports_vision=False,
            create_generator_fn=create_generator,
            get_runtime_load_info_fn=lambda _model_kit: {},
            tokenize_fn=lambda _model_kit, _prompt: [1],
        )

        with _running_server(runtime) as port:
            status, response_text = _request(
                port,
                "POST",
                "/v1/chat/completions",
                body=_request_body(),
            )

        assert status == 200
        assert _parse_sse(response_text) == [
            {"error": {"message": str(generation_error)}}
        ]


def test_mid_stream_generation_errors_emit_a_recognized_error_frame():
    generation_error = RuntimeError("generation failed after output")

    def create_generator(_model_kit, _prompt_tokens, **_kwargs):
        yield GenerationResult(
            text="partial output",
            tokens=[Token(id=10, text="partial output", logprob=-0.1)],
            top_logprobs=[],
            stop_condition=None,
        )
        raise generation_error

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        tokenize_fn=lambda _model_kit, _prompt: [1],
    )

    with _running_server(runtime) as port:
        status, response_text = _request(
            port,
            "POST",
            "/v1/chat/completions",
            body=_request_body(),
        )

    assert status == 200
    assert _parse_sse(response_text) == [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "partial output"},
                    "finish_reason": None,
                }
            ]
        },
        {"error": {"message": str(generation_error)}},
    ]
    assert "data: [DONE]" not in response_text


def test_stalled_sse_write_cancels_the_active_mlx_request(monkeypatch):
    monkeypatch.setattr(server_http, "_SSE_WRITE_TIMEOUT_SECONDS", 0.05)
    generation_stopped = threading.Event()
    stopped_request_ids = []
    large_text = "x" * (1024 * 1024)

    def create_generator(_model_kit, _prompt_tokens, **_kwargs):
        while not generation_stopped.is_set():
            yield GenerationResult(
                text=large_text,
                tokens=[],
                top_logprobs=[],
                stop_condition=None,
            )

    def stop_generation(_model_kit, request_id):
        stopped_request_ids.append(request_id)
        generation_stopped.set()

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        stop_generation_fn=stop_generation,
        tokenize_fn=lambda _model_kit, _prompt: [1, 2, 3],
    )

    with _running_server(runtime, send_buffer_size=4096) as port:
        encoded_body = json.dumps(_request_body()).encode("utf-8")
        with socket.socket() as client:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            client.settimeout(2)
            client.connect(("127.0.0.1", port))
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer secret-token\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(encoded_body)}\r\n\r\n".encode("ascii")
                + encoded_body
            )
            assert generation_stopped.wait(timeout=2)

    assert len(stopped_request_ids) == 1
    assert stopped_request_ids[0] != ""


def test_client_disconnect_stops_the_active_mlx_request():
    generation_stopped = threading.Event()
    stopped_request_ids = []

    def create_generator(_model_kit, _prompt_tokens, **kwargs):
        reporter = kwargs["prompt_progress_reporter"]
        reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=3,
            prefill_tokens_processed=0,
        )
        while not generation_stopped.is_set():
            reporter.update(is_draft=False, prefill_tokens_processed=1)
            time.sleep(0.01)
        yield GenerationResult(
            text="",
            tokens=[],
            top_logprobs=[],
            stop_condition=GenerationStopCondition(
                stop_reason="user_cancelled",
                stop_string="",
                stop_tokens=[],
            ),
        )

    def stop_generation(_model_kit, request_id):
        stopped_request_ids.append(request_id)
        generation_stopped.set()

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        create_generator_fn=create_generator,
        get_runtime_load_info_fn=lambda _model_kit: {},
        stop_generation_fn=stop_generation,
        tokenize_fn=lambda _model_kit, _prompt: [1, 2, 3],
    )

    with _running_server(runtime) as port:
        encoded_body = json.dumps(_request_body()).encode("utf-8")
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer secret-token\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(encoded_body)}\r\n\r\n".encode("ascii")
            + encoded_body
        )
        received = b""
        while b"\r\n\r\n" not in received:
            received += client.recv(4096)
        assert b"200 OK" in received
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()

        assert generation_stopped.wait(timeout=2)

    assert len(stopped_request_ids) == 1
    assert stopped_request_ids[0] != ""


def test_generation_session_cancellation_stops_the_exact_request():
    stopped_request_ids = []

    def stop_generation(_model_kit, request_id):
        stopped_request_ids.append(request_id)

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        stop_generation_fn=stop_generation,
    )
    session = GenerationSession(runtime)

    session.cancel()

    assert stopped_request_ids == [session.request_id]


def test_cancellation_failure_does_not_break_cleanup():
    stop_calls = []

    def stop_generation(_model_kit, request_id):
        stop_calls.append(request_id)
        raise RuntimeError("backend already stopped")

    runtime = EngineRuntime(
        _FakeModelKit(),
        supports_vision=False,
        stop_generation_fn=stop_generation,
    )
    session = GenerationSession(runtime)

    session.cancel()
    session.cancel()

    assert stop_calls == [session.request_id]


def test_runtime_unloads_model():
    unload_calls = []
    model_kit = _FakeModelKit()
    runtime = EngineRuntime(
        model_kit,
        supports_vision=False,
        unload_fn=lambda received_model_kit: unload_calls.append(received_model_kit),
    )

    runtime.unload()

    assert unload_calls == [model_kit]
