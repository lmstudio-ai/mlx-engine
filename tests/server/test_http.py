from contextlib import contextmanager
import http.client
import json
import socket
import struct
import threading
import time

from mlx_engine.server.http import (
    EngineRuntime,
    GenerationSession,
    MlxEngineHttpServer,
)
from mlx_engine.utils.generation_result import (
    GenerationResult,
    GenerationStopCondition,
)
from mlx_engine.utils.token import Token


class _FakeRenderer:
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
def _running_server(runtime):
    server = MlxEngineHttpServer(
        ("127.0.0.1", 0),
        api_key="secret-token",
        runtime=runtime,
    )
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
            str(500 * 1024 * 1024 + 1),
        )
        assert status == 413
        assert json.loads(body) == {
            "error": {"message": "Request body exceeds the 500 MiB limit."}
        }


def test_chat_stream_forwards_generation_settings_and_returns_usage():
    generation_calls = []

    def create_generator(model_kit, prompt_tokens, **kwargs):
        generation_calls.append((model_kit, prompt_tokens, kwargs))
        reporter = kwargs["prompt_progress_reporter"]
        assert reporter.begin(
            is_draft=False,
            cached_tokens=3,
            total_prompt_tokens=9,
            prefill_tokens_processed=0,
        )
        assert reporter.update(is_draft=False, prefill_tokens_processed=6)
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
            "prompt_tokens": 9,
            "completion_tokens": 2,
            "total_tokens": 11,
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


def test_generation_error_is_returned_inside_the_stream():
    def create_generator(_model_kit, _prompt_tokens, **_kwargs):
        raise RuntimeError("generation failed")
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
    assert _parse_sse(response_text) == [{"error": {"message": "generation failed"}}]


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
