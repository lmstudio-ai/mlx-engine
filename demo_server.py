import argparse
import json
import os
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from mlx_engine.generate import load_model, tokenize
from mlx_engine.batch_generate import BatchGenerationResult, create_batch_generator

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 128


def resolve_model_path(model_argument: str) -> str:
    if os.path.exists(model_argument):
        return model_argument

    local_paths = [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]

    for path in local_paths:
        candidate_path = os.path.join(path, model_argument)
        if os.path.exists(candidate_path):
            return candidate_path

    raise ValueError(f"Could not find model '{model_argument}' in local directories")


class RequestState:
    def __init__(self) -> None:
        self.request_slot_id: Optional[int] = None
        self.cached_tokens: Optional[int] = None
        self.total_prompt_tokens: Optional[int] = None
        self.result_queue: queue.Queue[Optional[BatchGenerationResult]] = queue.Queue()
        self.ready_event = threading.Event()
        self.error_message: Optional[str] = None


@dataclass
class PendingRequest:
    request_state: RequestState
    session_id: str
    prompt_tokens: list[int]
    max_tokens: int
    stop_strings: Optional[list[str]]
    top_logprobs: Optional[int]
    repetition_penalty: Optional[float]
    temp: Optional[float]
    top_p: Optional[float]
    min_p: Optional[float]
    top_k: Optional[int]
    seed: Optional[int]


class BatchScheduler:
    def __init__(self, batch_generator) -> None:
        self._batch_generator = batch_generator
        self._pending_requests: queue.Queue[PendingRequest] = queue.Queue()
        self._active_requests: dict[int, RequestState] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="mlx-batch-worker", daemon=True
        )

    def start(self) -> None:
        self._worker_thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self._worker_thread.join()
        self._batch_generator.close()

    def enqueue(self, pending_request: PendingRequest) -> RequestState:
        self._pending_requests.put(pending_request)
        ready = pending_request.request_state.ready_event.wait(timeout=30)
        if ready is False:
            pending_request.request_state.error_message = (
                "Timed out while enqueuing request"
            )
        return pending_request.request_state

    def _worker_loop(self) -> None:
        while self._shutdown_event.is_set() is False:
            pending_requests = self._drain_pending_requests()
            for pending_request in pending_requests:
                self._handle_pending_request(pending_request)

            if self._has_active_requests() is False:
                self._shutdown_event.wait(0.01)
                continue

            self._process_batch_results()

    def _drain_pending_requests(self) -> list[PendingRequest]:
        pending_requests: list[PendingRequest] = []
        try:
            pending_requests.append(self._pending_requests.get(timeout=0.01))
        except queue.Empty:
            return pending_requests

        while True:
            try:
                pending_requests.append(self._pending_requests.get_nowait())
            except queue.Empty:
                break
        return pending_requests

    def _handle_pending_request(self, pending_request: PendingRequest) -> None:
        request_state = pending_request.request_state
        try:
            (
                request_slot_id,
                cached_tokens,
                total_prompt_tokens,
            ) = self._batch_generator.insert(
                session_id=pending_request.session_id,
                prompt_tokens=pending_request.prompt_tokens,
                max_tokens=pending_request.max_tokens,
                stop_strings=pending_request.stop_strings,
                top_logprobs=pending_request.top_logprobs,
                repetition_penalty=pending_request.repetition_penalty,
                temp=pending_request.temp,
                top_p=pending_request.top_p,
                min_p=pending_request.min_p,
                top_k=pending_request.top_k,
                seed=pending_request.seed,
            )
            request_state.request_slot_id = request_slot_id
            request_state.cached_tokens = cached_tokens
            request_state.total_prompt_tokens = total_prompt_tokens
            with self._lock:
                self._active_requests[request_slot_id] = request_state
        except Exception as exception:
            request_state.error_message = str(exception)
        request_state.ready_event.set()

    def _has_active_requests(self) -> bool:
        with self._lock:
            return len(self._active_requests) > 0

    def _process_batch_results(self) -> None:
        try:
            batch_results = self._batch_generator.next()
        except Exception as exception:
            self._fail_active_requests(str(exception))
            return

        if len(batch_results) == 0:
            self._shutdown_event.wait(0.005)
            return

        for result in batch_results:
            request_state = self._get_request_state(result.request_slot_id)
            if request_state is None:
                continue
            request_state.result_queue.put(result)
            if result.stop_condition is not None:
                self._remove_request_state(result.request_slot_id)

    def _get_request_state(self, request_slot_id: int) -> Optional[RequestState]:
        with self._lock:
            return self._active_requests.get(request_slot_id)

    def _remove_request_state(self, request_slot_id: int) -> None:
        with self._lock:
            if request_slot_id in self._active_requests:
                del self._active_requests[request_slot_id]

    def _fail_active_requests(self, error_message: str) -> None:
        with self._lock:
            request_states = list(self._active_requests.values())
            self._active_requests.clear()
        for request_state in request_states:
            request_state.error_message = error_message
            request_state.result_queue.put(None)


class BatchHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        model_kit,
        batch_scheduler: BatchScheduler,
        default_max_tokens: int,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.model_kit = model_kit
        self.batch_scheduler = batch_scheduler
        self.default_max_tokens = default_max_tokens


class BatchRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json_response(404, {"error": "Not found"})
            return
        self._send_json_response(200, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json_response(404, {"error": "Not found"})
            return

        content_length = self._read_content_length()
        if content_length is None:
            return

        request_body = self.rfile.read(content_length)
        request_json = self._parse_json(request_body)
        if request_json is None:
            return

        prompt_value = request_json.get("prompt")
        if isinstance(prompt_value, str) is False or prompt_value == "":
            self._send_json_response(400, {"error": "prompt must be a non-empty string"})
            return

        session_id_value = request_json.get("session_id")
        if isinstance(session_id_value, str) is False:
            session_id_value = ""

        max_tokens_value = request_json.get("max_tokens")
        if isinstance(max_tokens_value, int) is False:
            max_tokens_value = self.server.default_max_tokens

        stop_strings_value = request_json.get("stop_strings")
        if isinstance(stop_strings_value, list) is False:
            stop_strings_value = None
        else:
            stop_strings_value = [
                stop_string
                for stop_string in stop_strings_value
                if isinstance(stop_string, str)
            ]

        top_logprobs_value = request_json.get("top_logprobs")
        if isinstance(top_logprobs_value, int) is False:
            top_logprobs_value = None

        repetition_penalty_value = request_json.get("repetition_penalty")
        if isinstance(repetition_penalty_value, (float, int)) is False:
            repetition_penalty_value = None
        else:
            repetition_penalty_value = float(repetition_penalty_value)

        temp_value = request_json.get("temp")
        if isinstance(temp_value, (float, int)) is False:
            temp_value = None
        else:
            temp_value = float(temp_value)

        top_p_value = request_json.get("top_p")
        if isinstance(top_p_value, (float, int)) is False:
            top_p_value = None
        else:
            top_p_value = float(top_p_value)

        min_p_value = request_json.get("min_p")
        if isinstance(min_p_value, (float, int)) is False:
            min_p_value = None
        else:
            min_p_value = float(min_p_value)

        top_k_value = request_json.get("top_k")
        if isinstance(top_k_value, int) is False:
            top_k_value = None

        seed_value = request_json.get("seed")
        if isinstance(seed_value, int) is False:
            seed_value = None

        prompt_tokens = tokenize(self.server.model_kit, prompt_value)
        pending_request = PendingRequest(
            request_state=RequestState(),
            session_id=session_id_value,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens_value,
            stop_strings=stop_strings_value,
            top_logprobs=top_logprobs_value,
            repetition_penalty=repetition_penalty_value,
            temp=temp_value,
            top_p=top_p_value,
            min_p=min_p_value,
            top_k=top_k_value,
            seed=seed_value,
        )
        request_state = self.server.batch_scheduler.enqueue(pending_request)
        if request_state.error_message is not None:
            self._send_json_response(500, {"error": request_state.error_message})
            return

        response_payload = self._collect_response_payload(request_state)
        if response_payload is None:
            return
        self._send_json_response(200, response_payload)

    def log_message(self, format, *arguments) -> None:
        return

    def _read_content_length(self) -> Optional[int]:
        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            self._send_json_response(411, {"error": "Missing Content-Length header"})
            return None
        try:
            return int(content_length_header)
        except ValueError:
            self._send_json_response(400, {"error": "Invalid Content-Length header"})
            return None

    def _parse_json(self, request_body: bytes) -> Optional[dict]:
        try:
            return json.loads(request_body)
        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON payload"})
            return None

    def _collect_response_payload(self, request_state: RequestState) -> Optional[dict]:
        collected_text_segments: list[str] = []
        token_count = 0
        stop_reason = None
        stop_string = None
        stop_tokens: list[int] = []

        while True:
            result = request_state.result_queue.get()
            if result is None:
                error_message = request_state.error_message or "Unknown error"
                self._send_json_response(500, {"error": error_message})
                return None
            if result.text != "":
                collected_text_segments.append(result.text)
            token_count += len(result.tokens)
            if result.stop_condition is not None:
                stop_reason = result.stop_condition.stop_reason
                stop_string = result.stop_condition.stop_string
                stop_tokens = result.stop_condition.stop_tokens
                break

        response_text = "".join(collected_text_segments)
        return {
            "request_slot_id": request_state.request_slot_id,
            "cached_tokens": request_state.cached_tokens,
            "total_prompt_tokens": request_state.total_prompt_tokens,
            "text": response_text,
            "token_count": token_count,
            "stop_reason": stop_reason,
            "stop_string": stop_string,
            "stop_tokens": stop_tokens,
        }

    def _send_json_response(self, status_code: int, payload: dict) -> None:
        response_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple HTTP server using mlx-engine continuous batching"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The file system path or model name to load",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, type=str)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument(
        "--completion-batch-size",
        default=8,
        type=int,
        help="Maximum number of active completion slots",
    )
    parser.add_argument(
        "--prefill-batch-size",
        default=2,
        type=int,
        help="Maximum number of prompts prefilling concurrently",
    )
    parser.add_argument(
        "--prefill-step-size",
        default=2048,
        type=int,
        help="Prompt prefill chunk size",
    )
    parser.add_argument("--default-max-tokens", default=DEFAULT_MAX_TOKENS, type=int)
    return parser


def main() -> None:
    parser = setup_arg_parser()
    arguments = parser.parse_args()

    model_path = resolve_model_path(arguments.model)
    print("Loading model...", flush=True)
    model_kit = load_model(
        model_path=model_path,
        max_kv_size=arguments.max_kv_size,
    )
    print("Model load complete", flush=True)

    batch_generator = create_batch_generator(
        model_kit=model_kit,
        completion_batch_size=arguments.completion_batch_size,
        prefill_batch_size=arguments.prefill_batch_size,
        prefill_step_size=arguments.prefill_step_size,
    )
    batch_scheduler = BatchScheduler(batch_generator)
    batch_scheduler.start()

    server = BatchHTTPServer(
        (arguments.host, arguments.port),
        BatchRequestHandler,
        model_kit=model_kit,
        batch_scheduler=batch_scheduler,
        default_max_tokens=arguments.default_max_tokens,
    )

    print(f"Listening on http://{arguments.host}:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        batch_scheduler.shutdown()


if __name__ == "__main__":
    main()
