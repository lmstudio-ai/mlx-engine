import json
import logging
import os
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.server import LRUPromptCache
from mlx_lm.utils import _download, load_model, load_tokenizer
from mlx.utils import tree_flatten

from mlx_engine.model_kit.batched_model_kit import _prepare_prompt_cache_for_generation
from mlx_engine.model_kit.batched_model_kit_types import (
    BatchedGenerationResponse,
    CancelGenerationRequest,
    RequestCancelled,
)
from mlx_engine.utils.disable_hf_download import _original_snapshot_download
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_engine.utils.mlx_lm_stream import (
    log_mlx_stream_state,
    prepare_mlx_lm_generation_stream,
)
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter
from mlx_engine.utils.token import Token


logger = logging.getLogger(__name__)

SCHEDULER_PROTOCOL_VERSION = 1
SCHEDULER_MESSAGE_GENERATE = "generate"
SCHEDULER_MESSAGE_CANCEL = "cancel"
SCHEDULER_MESSAGE_SHUTDOWN = "shutdown"
SCHEDULER_GENERATION_STEPS_PER_TICK = 1


@dataclass
class DistributedSchedulerGenerationRequest:
    response_queue: Queue | None
    prompt_tokens: list[int]
    request_id: str
    sampler: Any
    logits_processors: list[Any]
    top_logprobs: int
    max_tokens: int
    sampling: dict[str, Any]
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[int]
    min_tokens_to_keep: Optional[int]


@dataclass
class DistributedSchedulerShutdownRequest:
    pass


@dataclass
class DistributedModelThreadRequest:
    description: str
    callback: Any
    response_queue: Queue
    stream_results: bool
    created_at: float
    caller_thread_name: str
    caller_thread_ident: int | None
    caller_stopped: threading.Event | None


def _format_size_bytes(size_bytes: int) -> str:
    gibibytes = size_bytes / (1024 * 1024 * 1024)
    return f"{gibibytes:.2f} GiB"


def _current_thread_summary() -> str:
    current_thread = threading.current_thread()
    return f"{current_thread.name}:{current_thread.ident}"


def _model_snapshot_summary(model_path: Path) -> str:
    try:
        safetensor_files = list(model_path.glob("*.safetensors"))
        total_safetensor_size = sum(
            safetensor_file.stat().st_size for safetensor_file in safetensor_files
        )
        return (
            f"safetensors={len(safetensor_files)} "
            f"safetensorsSize={_format_size_bytes(total_safetensor_size)}"
        )
    except Exception as caught_error:
        return f"snapshotSummaryError={caught_error}"


def _instrumented_tensor_sharded_load(
    repo: str | Path,
    tensor_group: Any,
) -> tuple[Any, Any]:
    rank = tensor_group.rank()
    size = tensor_group.size()
    started_at = time.monotonic()
    log_mlx_stream_state(
        reason="sharded-load-start",
        distributed_group=tensor_group,
        details=f"repo={repo}",
    )

    logger.info("[sharded_load] rank %s/%s resolving metadata files", rank, size)
    model_path = _download(
        repo,
        allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
    )
    logger.info(
        "[sharded_load] rank %s/%s metadata resolved to %s after %.1fs",
        rank,
        size,
        model_path,
        time.monotonic() - started_at,
    )

    logger.info("[sharded_load] rank %s/%s lazy-loading model skeleton", rank, size)
    model, config = load_model(model_path, lazy=True, strict=False)
    logger.info(
        "[sharded_load] rank %s/%s model skeleton loaded after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )

    has_tensor_parallel = hasattr(model, "shard")
    logger.info(
        "[sharded_load] rank %s/%s tensorParallel=%s",
        rank,
        size,
        has_tensor_parallel,
    )
    if not has_tensor_parallel:
        raise ValueError("The model does not support tensor parallelism")

    logger.info("[sharded_load] rank %s/%s ensuring weight files", rank, size)
    _download(repo)
    logger.info(
        "[sharded_load] rank %s/%s weight files available after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )

    logger.info("[sharded_load] rank %s/%s loading tokenizer", rank, size)
    tokenizer = load_tokenizer(
        model_path,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id", None),
    )
    logger.info(
        "[sharded_load] rank %s/%s tokenizer loaded after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )

    logger.info("[sharded_load] rank %s/%s re-loading model for weights", rank, size)
    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.info(
        "[sharded_load] rank %s/%s weight model loaded lazily after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )

    logger.info("[sharded_load] rank %s/%s applying tensor shard", rank, size)
    model.shard(tensor_group)
    logger.info(
        "[sharded_load] rank %s/%s tensor shard applied after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )

    log_mlx_stream_state(
        reason="sharded-load-before-parameter-eval",
        distributed_group=tensor_group,
    )
    logger.info("[sharded_load] rank %s/%s evaluating model parameters", rank, size)
    _evaluate_model_parameters_with_trace(model, rank, size, started_at)
    logger.info(
        "[sharded_load] rank %s/%s model parameters evaluated after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )
    log_mlx_stream_state(
        reason="sharded-load-after-parameter-eval",
        distributed_group=tensor_group,
    )

    logger.info("[sharded_load] rank %s/%s synchronizing all_sum", rank, size)
    log_mlx_stream_state(
        reason="sharded-load-before-all-sum",
        distributed_group=tensor_group,
    )
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    logger.info(
        "[sharded_load] rank %s/%s all_sum synchronized after %.1fs",
        rank,
        size,
        time.monotonic() - started_at,
    )
    log_mlx_stream_state(
        reason="sharded-load-after-all-sum",
        distributed_group=tensor_group,
    )

    return model, tokenizer


def _evaluate_model_parameters_with_trace(
    model: Any,
    rank: int,
    size: int,
    started_at: float,
) -> None:
    parameters = tree_flatten(model.parameters())
    logger.info(
        "[sharded_load] rank %s/%s evaluating %s model parameters one-by-one",
        rank,
        size,
        len(parameters),
    )
    for parameter_index, (parameter_name, parameter) in enumerate(parameters):
        parameter_started_at = time.monotonic()
        logger.info(
            "[sharded_load] rank %s/%s evaluating parameter %s/%s %s shape=%s dtype=%s",
            rank,
            size,
            parameter_index + 1,
            len(parameters),
            parameter_name,
            getattr(parameter, "shape", "<unknown>"),
            getattr(parameter, "dtype", "<unknown>"),
        )
        mx.eval(parameter)
        logger.info(
            "[sharded_load] rank %s/%s evaluated parameter %s/%s %s after %.1fs total=%.1fs",
            rank,
            size,
            parameter_index + 1,
            len(parameters),
            parameter_name,
            time.monotonic() - parameter_started_at,
            time.monotonic() - started_at,
        )


class DistributedModelKit:
    """
    Minimal text-only model kit for MLX distributed tensor parallel inference.

    This is intentionally smaller than ModelKit. It exists so cluster-tool can
    validate distributed loading and generation through mlx-engine before that
    path is wired into the production app runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        prefill_step_size: int,
        *,
        max_kv_size: int | None = None,
        max_seq_nums: int | None = None,
        trust_remote_code: bool = False,
        distributed_group: Any = None,
    ):
        self.generation_lock = threading.Lock()
        self.pending_requests: dict[str, threading.Event] = {}
        self._requests: Queue = Queue()
        self._prompt_cache = LRUPromptCache()
        self._batch_results: dict[int, dict[str, Any]] = {}
        self._backend_exception: Exception | None = None
        self._generation_thread: threading.Thread | None = None
        self._model_thread: threading.Thread | None = None
        self._model_thread_ident: int | None = None
        self._model_thread_requests: Queue | None = None
        self._shutdown = threading.Event()
        self.prefill_step_size = prefill_step_size
        self.max_kv_size = max_kv_size
        self.max_seq_nums = 1 if max_seq_nums is None or max_seq_nums < 1 else max_seq_nums
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.draft_model = None
        self._cross_prompt_cache_active = False

        logger.info("Resolving distributed model path from %s", model_path)
        self.model_path = self._resolve_model_path(model_path)
        logger.info(
            "Resolved distributed model path to %s (%s)",
            self.model_path,
            _model_snapshot_summary(self.model_path),
        )
        config_json = json.loads((self.model_path / "config.json").read_text())
        if "vision_config" in config_json:
            logger.info(
                "DistributedModelKit detected vision_config; loading through the distributed text-only MLX-LM path"
            )
        self.model_type = config_json.get("model_type", None)
        logger.info(
            "Loaded distributed model config model_type=%s max_kv_size=%s max_seq_nums=%s prefill_step_size=%s",
            self.model_type,
            self.max_kv_size,
            self.max_seq_nums,
            self.prefill_step_size,
        )

        self.group = distributed_group
        if self.group is None:
            logger.info("Initializing MLX distributed group inside DistributedModelKit")
            self.group = mx.distributed.init()
        if self.group.size() <= 1:
            raise ValueError("DistributedModelKit requires more than one MLX rank")

        if self.uses_distributed_batching():
            self._load_model_shard()
        else:
            self._start_model_thread()
            self._run_on_model_thread_sync(
                "distributed-model-load",
                self._load_model_shard,
            )

    def _load_model_shard(self) -> None:
        logger.info(
            "Loading distributed model shard from %s on rank %s/%s...",
            self.model_path,
            self.group.rank(),
            self.group.size(),
        )
        sharded_load_stream = prepare_mlx_lm_generation_stream(
            reason="distributed-sharded-load",
            distributed_group=self.group,
            use_default_stream=True,
        )
        log_mlx_stream_state(
            reason="distributed-model-load-start",
            distributed_group=self.group,
            details=f"model_path={self.model_path}",
        )
        sharded_load_started_at = time.monotonic()
        with mx.stream(sharded_load_stream):
            self.model, self.tokenizer = _instrumented_tensor_sharded_load(
                self.model_path,
                tensor_group=self.group,
            )
        log_mlx_stream_state(
            reason="distributed-model-load-finished",
            distributed_group=self.group,
            details=f"model_path={self.model_path}",
        )
        logger.info(
            "sharded_load returned on rank %s/%s after %.1fs",
            self.group.rank(),
            self.group.size(),
            time.monotonic() - sharded_load_started_at,
        )
        logger.info("Applying tokenizer fixes on rank %s", self.group.rank())
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer,
            model_path=self.model_path,
            model_type=self.model_type,
        )
        self.detokenizer = self.tokenizer.detokenizer
        logger.info("Distributed model shard loaded on rank %s", self.group.rank())

    def _start_model_thread(self) -> None:
        if self._model_thread is not None:
            return

        logger.info(
            "Starting distributed model thread on rank %s/%s caller_thread=%s",
            self.group.rank(),
            self.group.size(),
            _current_thread_summary(),
        )
        self._model_thread_requests = Queue()
        self._model_thread = threading.Thread(
            target=self._model_thread_loop,
            daemon=True,
            name="mlx-distributed-model-thread",
        )
        self._model_thread.start()
        logger.info(
            "Started distributed model thread on rank %s/%s thread_name=%s",
            self.group.rank(),
            self.group.size(),
            self._model_thread.name,
        )

    def _model_thread_loop(self) -> None:
        self._model_thread_ident = threading.get_ident()
        log_mlx_stream_state(
            reason="distributed-model-thread-started",
            distributed_group=self.group,
        )
        assert self._model_thread_requests is not None

        while True:
            request = self._model_thread_requests.get()
            if request is None:
                log_mlx_stream_state(
                    reason="distributed-model-thread-stopping",
                    distributed_group=self.group,
                )
                logger.info(
                    "Distributed model thread stopping on rank %s/%s thread=%s",
                    self.group.rank(),
                    self.group.size(),
                    _current_thread_summary(),
                )
                return
            request_started_at = time.monotonic()
            caller_wait_seconds = request_started_at - request.created_at
            try:
                logger.info(
                    "Distributed model thread dequeued %s on rank %s/%s "
                    "thread=%s caller_thread=%s caller_ident=%s caller_wait=%.3fs pending=%s",
                    request.description,
                    self.group.rank(),
                    self.group.size(),
                    _current_thread_summary(),
                    request.caller_thread_name,
                    request.caller_thread_ident,
                    caller_wait_seconds,
                    self._model_thread_requests.qsize(),
                )
                log_mlx_stream_state(
                    reason="distributed-model-thread-before-callback",
                    distributed_group=self.group,
                    details=(
                        f"description={request.description} "
                        f"stream_results={request.stream_results} "
                        f"caller_wait={caller_wait_seconds:.3f}s"
                    ),
                )
                if request.stream_results:
                    item_count = 0
                    stream_completed = False
                    if (
                        request.caller_stopped is not None
                        and request.caller_stopped.is_set()
                    ):
                        logger.info(
                            "Distributed model thread skipping stream for %s on rank %s/%s "
                            "because caller stopped before callback elapsed=%.3fs",
                            request.description,
                            self.group.rank(),
                            self.group.size(),
                            time.monotonic() - request_started_at,
                        )
                        continue
                    result = request.callback()
                    try:
                        for item in result:
                            if (
                                request.caller_stopped is not None
                                and request.caller_stopped.is_set()
                            ):
                                logger.info(
                                    "Distributed model thread stopping stream for %s on rank %s/%s "
                                    "before delivering item after caller stopped items=%s elapsed=%.3fs",
                                    request.description,
                                    self.group.rank(),
                                    self.group.size(),
                                    item_count,
                                    time.monotonic() - request_started_at,
                                )
                                break
                            if item_count == 0:
                                logger.info(
                                    "Distributed model thread received first stream item for %s "
                                    "on rank %s/%s after %.3fs item_type=%s",
                                    request.description,
                                    self.group.rank(),
                                    self.group.size(),
                                    time.monotonic() - request_started_at,
                                    type(item).__name__,
                                )
                            item_count += 1
                            if not self._put_model_thread_stream_response(
                                request=request,
                                item=item,
                                item_count=item_count,
                                request_started_at=request_started_at,
                            ):
                                break
                            if (
                                request.caller_stopped is not None
                                and request.caller_stopped.is_set()
                            ):
                                logger.info(
                                    "Distributed model thread stopping stream for %s on rank %s/%s "
                                    "after caller stopped items=%s elapsed=%.3fs",
                                    request.description,
                                    self.group.rank(),
                                    self.group.size(),
                                    item_count,
                                    time.monotonic() - request_started_at,
                                )
                                break
                        else:
                            stream_completed = True
                    finally:
                        close_result = getattr(result, "close", None)
                        if callable(close_result):
                            close_result()
                    if stream_completed:
                        logger.info(
                            "Distributed model thread completed stream for %s on rank %s/%s "
                            "items=%s elapsed=%.3fs",
                            request.description,
                            self.group.rank(),
                            self.group.size(),
                            item_count,
                            time.monotonic() - request_started_at,
                        )
                        self._put_model_thread_stream_response(
                            request=request,
                            item=None,
                            item_count=item_count,
                            request_started_at=request_started_at,
                        )
                else:
                    result = request.callback()
                    logger.info(
                        "Distributed model thread completed sync request %s on rank %s/%s "
                        "elapsed=%.3fs result_type=%s",
                        request.description,
                        self.group.rank(),
                        self.group.size(),
                        time.monotonic() - request_started_at,
                        type(result).__name__,
                    )
                    request.response_queue.put((True, result))
            except Exception as caught_error:
                logger.exception(
                    "Distributed model thread failed during %s on rank %s/%s "
                    "elapsed=%.3fs stream_results=%s",
                    request.description,
                    self.group.rank(),
                    self.group.size(),
                    time.monotonic() - request_started_at,
                    request.stream_results,
                )
                if request.stream_results:
                    self._put_model_thread_stream_response(
                        request=request,
                        item=caught_error,
                        item_count=0,
                        request_started_at=request_started_at,
                    )
                else:
                    request.response_queue.put((False, caught_error))

    def _put_model_thread_stream_response(
        self,
        *,
        request: DistributedModelThreadRequest,
        item: Any,
        item_count: int,
        request_started_at: float,
    ) -> bool:
        while True:
            if request.caller_stopped is not None and request.caller_stopped.is_set():
                logger.info(
                    "Distributed model thread stopping stream for %s on rank %s/%s "
                    "while waiting for caller items=%s elapsed=%.3fs pending_response=%s",
                    request.description,
                    self.group.rank(),
                    self.group.size(),
                    item_count,
                    time.monotonic() - request_started_at,
                    type(item).__name__,
                )
                return False
            if self._shutdown.is_set():
                logger.info(
                    "Distributed model thread stopping stream for %s on rank %s/%s "
                    "because shutdown is set items=%s elapsed=%.3fs pending_response=%s",
                    request.description,
                    self.group.rank(),
                    self.group.size(),
                    item_count,
                    time.monotonic() - request_started_at,
                    type(item).__name__,
                )
                return False
            try:
                request.response_queue.put(item, timeout=0.1)
                return True
            except QueueFull:
                continue

    def _run_on_model_thread_sync(self, description: str, callback: Any) -> Any:
        if self._model_thread_requests is None:
            logger.info(
                "Running distributed model request inline because model thread is disabled: "
                "%s rank=%s/%s thread=%s",
                description,
                self.group.rank(),
                self.group.size(),
                _current_thread_summary(),
            )
            return callback()
        if threading.get_ident() == self._model_thread_ident:
            logger.info(
                "Running distributed model request directly on model thread: "
                "%s rank=%s/%s thread=%s",
                description,
                self.group.rank(),
                self.group.size(),
                _current_thread_summary(),
            )
            return callback()

        current_thread = threading.current_thread()
        created_at = time.monotonic()
        logger.info(
            "Queueing distributed model sync request %s on rank %s/%s "
            "caller_thread=%s caller_ident=%s model_thread_ident=%s",
            description,
            self.group.rank(),
            self.group.size(),
            current_thread.name,
            current_thread.ident,
            self._model_thread_ident,
        )
        log_mlx_stream_state(
            reason="distributed-model-thread-sync-queued",
            distributed_group=self.group,
            details=(
                f"description={description} "
                f"model_thread_ident={self._model_thread_ident}"
            ),
        )
        response_queue: Queue = Queue()
        self._model_thread_requests.put(
            DistributedModelThreadRequest(
                description=description,
                callback=callback,
                response_queue=response_queue,
                stream_results=False,
                created_at=created_at,
                caller_thread_name=current_thread.name,
                caller_thread_ident=current_thread.ident,
                caller_stopped=None,
            )
        )
        success, value = response_queue.get()
        logger.info(
            "Distributed model sync request returned %s on rank %s/%s "
            "success=%s elapsed=%.3fs value_type=%s",
            description,
            self.group.rank(),
            self.group.size(),
            success,
            time.monotonic() - created_at,
            type(value).__name__,
        )
        if success:
            return value
        raise value

    def run_generator_on_model_thread(
        self,
        *,
        description: str,
        callback: Any,
    ) -> Iterable[Any]:
        if self._model_thread_requests is None:
            logger.info(
                "Running distributed generator inline because model thread is disabled: "
                "%s rank=%s/%s thread=%s",
                description,
                self.group.rank(),
                self.group.size(),
                _current_thread_summary(),
            )
            yield from callback()
            return
        if threading.get_ident() == self._model_thread_ident:
            logger.info(
                "Running distributed generator directly on model thread: "
                "%s rank=%s/%s thread=%s",
                description,
                self.group.rank(),
                self.group.size(),
                _current_thread_summary(),
            )
            yield from callback()
            return

        current_thread = threading.current_thread()
        created_at = time.monotonic()
        logger.info(
            "Queueing distributed model streaming request %s on rank %s/%s "
            "caller_thread=%s caller_ident=%s model_thread_ident=%s",
            description,
            self.group.rank(),
            self.group.size(),
            current_thread.name,
            current_thread.ident,
            self._model_thread_ident,
        )
        log_mlx_stream_state(
            reason="distributed-model-thread-stream-queued",
            distributed_group=self.group,
            details=(
                f"description={description} "
                f"model_thread_ident={self._model_thread_ident}"
            ),
        )
        response_queue: Queue = Queue(maxsize=1)
        caller_stopped = threading.Event()
        self._model_thread_requests.put(
            DistributedModelThreadRequest(
                description=description,
                callback=callback,
                response_queue=response_queue,
                stream_results=True,
                created_at=created_at,
                caller_thread_name=current_thread.name,
                caller_thread_ident=current_thread.ident,
                caller_stopped=caller_stopped,
            )
        )

        item_count = 0
        stream_finished = False
        stream_failed = False
        try:
            while True:
                try:
                    item = response_queue.get(timeout=0.1)
                except QueueEmpty:
                    if self._shutdown.is_set():
                        stream_finished = True
                        logger.info(
                            "Distributed model streaming request stopped by shutdown %s on rank %s/%s "
                            "items=%s elapsed=%.3fs",
                            description,
                            self.group.rank(),
                            self.group.size(),
                            item_count,
                            time.monotonic() - created_at,
                        )
                        return
                    continue
                if item is None:
                    stream_finished = True
                    logger.info(
                        "Distributed model streaming request completed %s on rank %s/%s "
                        "items=%s elapsed=%.3fs",
                        description,
                        self.group.rank(),
                        self.group.size(),
                        item_count,
                        time.monotonic() - created_at,
                    )
                    return
                if isinstance(item, Exception):
                    stream_failed = True
                    logger.error(
                        "Distributed model streaming request failed %s on rank %s/%s "
                        "items=%s elapsed=%.3fs error_type=%s error=%r",
                        description,
                        self.group.rank(),
                        self.group.size(),
                        item_count,
                        time.monotonic() - created_at,
                        type(item).__name__,
                        item,
                    )
                    raise item
                if item_count == 0:
                    logger.info(
                        "Distributed model streaming request yielded first item %s on rank %s/%s "
                        "after %.3fs item_type=%s",
                        description,
                        self.group.rank(),
                        self.group.size(),
                        time.monotonic() - created_at,
                        type(item).__name__,
                    )
                item_count += 1
                yield item
        finally:
            if not stream_finished and not stream_failed:
                caller_stopped.set()
                logger.info(
                    "Distributed model streaming request caller stopped %s on rank %s/%s "
                    "items=%s elapsed=%.3fs",
                    description,
                    self.group.rank(),
                    self.group.size(),
                    item_count,
                    time.monotonic() - created_at,
                )

    def _resolve_model_path(self, model_path: str | Path) -> Path:
        candidate_path = Path(model_path).expanduser()
        if candidate_path.exists():
            return candidate_path

        try:
            return Path(
                _original_snapshot_download(
                    str(model_path),
                    local_files_only=True,
                )
            )
        except Exception as caught_error:
            raise FileNotFoundError(
                f"Model is not available in the local Hugging Face cache: {model_path}"
            ) from caught_error

    def start(self):
        if not self.uses_distributed_batching():
            self._run_on_model_thread_sync(
                "distributed-model-start",
                self._start_on_current_thread,
            )
            return
        self._start_on_current_thread()

    def _start_on_current_thread(self) -> None:
        log_mlx_stream_state(
            reason="distributed-model-start-before-synchronize",
            distributed_group=self.group,
        )
        logger.info(
            "Synchronizing distributed model start on rank %s/%s",
            self.group.rank(),
            self.group.size(),
        )
        mx.synchronize()
        log_mlx_stream_state(
            reason="distributed-model-start-after-synchronize",
            distributed_group=self.group,
        )
        if self.uses_distributed_batching():
            seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
            mx.random.seed(seed)
            logger.info(
                "Distributed batched scheduler seed synchronized on rank %s/%s",
                self.group.rank(),
                self.group.size(),
            )
            if self.group.rank() == 0:
                self._generation_thread = threading.Thread(
                    target=self._generate_with_exception_handling,
                    daemon=True,
                    name="mlx-distributed-batched-scheduler",
                )
                self._generation_thread.start()
        logger.info("Distributed model start synchronized on rank %s", self.group.rank())

    def uses_distributed_batching(self) -> bool:
        return self.max_seq_nums > 1

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_reporter: PromptProgressReporter,
        generate_args: dict,
        max_image_size: tuple[int, int] | None,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        if images_b64 is not None and len(images_b64) > 0:
            raise ValueError("DistributedModelKit does not support images yet")
        if speculative_decoding_toggle is True:
            raise ValueError("DistributedModelKit does not support draft models yet")
        if len(prompt_tokens) == 0:
            logger.warning(
                "Received empty prompt. Generation quality will likely be poor"
            )
            prompt_tokens = self.tokenize(" ")
        return mx.array(prompt_tokens), None

    def is_cross_prompt_cache_active(self) -> bool:
        return self._cross_prompt_cache_active

    def record_token_to_cache(self, token: int) -> None:
        return

    def is_draft_model_compatible(self, path: str) -> bool:
        return False

    def load_draft_model(self, path: str) -> None:
        raise ValueError("DistributedModelKit does not support draft models")

    def unload_draft_model(self) -> None:
        return

    def generate(
        self,
        *,
        prompt_tokens,
        request_id,
        sampler,
        logits_processors,
        prompt_progress_callback,
        top_logprobs,
        max_tokens,
        sampling: dict[str, Any],
        repetition_penalty: Optional[float],
        repetition_context_size: Optional[int],
        min_tokens_to_keep: Optional[int],
    ) -> Iterable[BatchedGenerationResponse]:
        if not self.uses_distributed_batching():
            raise RuntimeError("Distributed batched generation is not enabled")
        if self._shutdown.is_set():
            raise RuntimeError("Cannot accept new requests when model is shutdown")
        if isinstance(self._backend_exception, Exception):
            raise self._backend_exception
        if self.group.rank() != 0:
            raise RuntimeError("Only distributed rank 0 can accept generation requests")

        response_queue: Queue = Queue()
        self._requests.put(
            DistributedSchedulerGenerationRequest(
                response_queue=response_queue,
                prompt_tokens=list(prompt_tokens),
                request_id=request_id,
                sampler=sampler,
                logits_processors=logits_processors,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                sampling=sampling,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                min_tokens_to_keep=min_tokens_to_keep,
            )
        )

        def _inner() -> Iterable[BatchedGenerationResponse]:
            while True:
                try:
                    response = response_queue.get_nowait()
                except QueueEmpty:
                    if isinstance(self._backend_exception, Exception):
                        raise self._backend_exception
                    if self._shutdown.is_set():
                        raise RequestCancelled("Model shutdown requested")
                    time.sleep(0.001)
                    continue
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if prompt_progress_callback is not None:
                        prompt_progress_callback(*response)
                    continue
                yield response

        return _inner()

    def remove(self, request_id: str):
        if self.uses_distributed_batching():
            self._requests.put(CancelGenerationRequest(request_id))
            return
        self.cancel_request(request_id)

    def cancel_request(self, request_id: str) -> bool:
        if request_id in self.pending_requests:
            logger.warning(
                "Ignoring local cancellation for distributed request %s. "
                "Distributed ranks must either drain the request together or be "
                "torn down by the rank supervisor.",
                request_id,
            )
        return False

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        if self.uses_distributed_batching() and self.group.rank() == 0:
            self._requests.put(DistributedSchedulerShutdownRequest())
            if self._generation_thread is not None:
                self._generation_thread.join()
            self._shutdown.set()
            return
        self._shutdown.set()
        if self._model_thread_requests is not None:
            logger.info(
                "Requesting distributed model thread shutdown on rank %s/%s caller_thread=%s",
                self.group.rank(),
                self.group.size(),
                _current_thread_summary(),
            )
            self._model_thread_requests.put(None)
        if (
            self._model_thread is not None
            and threading.get_ident() != self._model_thread_ident
        ):
            join_started_at = time.monotonic()
            self._model_thread.join(timeout=5)
            logger.info(
                "Distributed model thread join finished on rank %s/%s "
                "elapsed=%.3fs alive=%s",
                self.group.rank(),
                self.group.size(),
                time.monotonic() - join_started_at,
                self._model_thread.is_alive(),
            )

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def run_worker_loop(self) -> None:
        if not self.uses_distributed_batching():
            raise RuntimeError("Distributed batched worker loop is not enabled")
        if self.group.rank() == 0:
            raise RuntimeError("Distributed batched worker loop cannot run on rank 0")
        logger.info(
            "Distributed batched worker rank %s/%s entering scheduler loop",
            self.group.rank(),
            self.group.size(),
        )
        self._generate_with_exception_handling()

    def _generate_with_exception_handling(self) -> None:
        try:
            self._generate()
        except Exception:
            error_string = (
                "Encountered fatal exception in distributed batched scheduler: "
                f"{traceback.format_exc()}"
            )
            logger.error(error_string)
            self._backend_exception = Exception(error_string)
            for entry in self._batch_results.values():
                response_queue = entry.get("response_queue")
                if response_queue is not None:
                    response_queue.put(self._backend_exception)
            time.sleep(1)
            logging.shutdown()
            os._exit(1)

    def _get_local_scheduler_item(
        self, timeout: None | float
    ) -> DistributedSchedulerGenerationRequest | CancelGenerationRequest | DistributedSchedulerShutdownRequest | None:
        try:
            if timeout is not None:
                return self._requests.get(timeout=timeout)
            return self._requests.get_nowait()
        except QueueEmpty:
            return None

    def _next_scheduler_item(
        self, timeout: None | float
    ) -> DistributedSchedulerGenerationRequest | CancelGenerationRequest | DistributedSchedulerShutdownRequest | None:
        local_item = None
        if self.group.rank() == 0:
            local_item = self._get_local_scheduler_item(timeout)
            if isinstance(local_item, DistributedSchedulerGenerationRequest):
                logger.info(
                    "Distributed batched rank 0 sharing generate request_id=%s max_tokens=%s prompt_tokens=%s",
                    local_item.request_id,
                    local_item.max_tokens,
                    len(local_item.prompt_tokens),
                )
            elif isinstance(local_item, CancelGenerationRequest):
                logger.info(
                    "Distributed batched rank 0 sharing cancel request_id=%s",
                    local_item.request_id,
                )
            elif isinstance(local_item, DistributedSchedulerShutdownRequest):
                logger.info("Distributed batched rank 0 sharing shutdown")
            self._share_scheduler_message(self._scheduler_item_to_message(local_item))
            return local_item

        message = self._share_scheduler_message(None)
        if message is not None:
            logger.info(
                "Distributed batched worker rank %s/%s received scheduler message type=%s request_id=%s",
                self.group.rank(),
                self.group.size(),
                message.get("type"),
                message.get("requestId"),
            )
        return self._scheduler_message_to_worker_item(message)

    def _scheduler_item_to_message(
        self,
        item: DistributedSchedulerGenerationRequest
        | CancelGenerationRequest
        | DistributedSchedulerShutdownRequest
        | None,
    ) -> dict[str, Any] | None:
        if item is None:
            return None
        if isinstance(item, DistributedSchedulerShutdownRequest):
            return {
                "version": SCHEDULER_PROTOCOL_VERSION,
                "type": SCHEDULER_MESSAGE_SHUTDOWN,
            }
        if isinstance(item, CancelGenerationRequest):
            return {
                "version": SCHEDULER_PROTOCOL_VERSION,
                "type": SCHEDULER_MESSAGE_CANCEL,
                "requestId": item.request_id,
            }
        return {
            "version": SCHEDULER_PROTOCOL_VERSION,
            "type": SCHEDULER_MESSAGE_GENERATE,
            "requestId": item.request_id,
            "promptTokens": item.prompt_tokens,
            "maxTokens": item.max_tokens,
            "sampling": item.sampling,
            "repetitionPenalty": item.repetition_penalty,
            "repetitionContextSize": item.repetition_context_size,
            "minTokensToKeep": item.min_tokens_to_keep,
        }

    def _scheduler_message_to_worker_item(
        self, message: dict[str, Any] | None
    ) -> DistributedSchedulerGenerationRequest | CancelGenerationRequest | DistributedSchedulerShutdownRequest | None:
        if message is None:
            return None
        message_type = message["type"]
        if message_type == SCHEDULER_MESSAGE_SHUTDOWN:
            return DistributedSchedulerShutdownRequest()
        if message_type == SCHEDULER_MESSAGE_CANCEL:
            return CancelGenerationRequest(message["requestId"])
        if message_type != SCHEDULER_MESSAGE_GENERATE:
            raise ValueError(f"Unsupported distributed scheduler message type {message_type}")

        sampling = message["sampling"]
        from mlx_engine.utils.generation_helpers import (
            create_sampler,
            setup_logits_processors,
            setup_repetition_penalty,
        )

        repetition_penalty = message["repetitionPenalty"]
        repetition_context_size = message["repetitionContextSize"]
        repetition_penalty_kwargs = setup_repetition_penalty(
            repetition_penalty, repetition_context_size
        )
        prompt_tokens = message["promptTokens"]
        return DistributedSchedulerGenerationRequest(
            response_queue=None,
            prompt_tokens=prompt_tokens,
            request_id=message["requestId"],
            sampler=create_sampler(
                sampling["temperature"],
                sampling["topP"],
                sampling["minP"],
                message["minTokensToKeep"],
                sampling["topK"],
            ),
            logits_processors=setup_logits_processors(
                repetition_penalty,
                repetition_penalty_kwargs,
                prompt_tokens,
                prompt_tokens,
                None,
                self.tokenizer,
            ),
            top_logprobs=0,
            max_tokens=message["maxTokens"],
            sampling=sampling,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            min_tokens_to_keep=message["minTokensToKeep"],
        )

    def _share_scheduler_message(
        self, message: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if self.group.rank() == 0:
            if message is None:
                mx.eval(mx.distributed.all_sum(0))
                return None
            encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
            data = mx.array(list(encoded), dtype=mx.uint8)
            mx.eval(mx.distributed.all_sum(data.size))
            mx.eval(mx.distributed.all_sum(data))
            return message

        data_size = mx.distributed.all_sum(0).item()
        if data_size == 0:
            return None
        data = mx.zeros(data_size, dtype=mx.uint8)
        data = mx.distributed.all_sum(data)
        return json.loads(bytes(data.tolist()).decode("utf-8"))

    def _cancel_scheduler_request(self, batch_generator: BatchGenerator, request_id: str) -> None:
        found_request_id = False
        for uid, entry in list(self._batch_results.items()):
            if entry.get("request_id") == request_id:
                found_request_id = True
                batch_generator.remove([uid])
                response_queue = entry.get("response_queue")
                if response_queue is not None:
                    response_queue.put(RequestCancelled())
                del self._batch_results[uid]
                break
        if not found_request_id:
            logger.warning("Could not cancel request_id=%s (id not found)", request_id)

    def _insert_scheduler_request(
        self,
        batch_generator: BatchGenerator,
        current_model_key: str,
        request: DistributedSchedulerGenerationRequest,
    ) -> None:
        cache, cached_prefix, rest = _prepare_prompt_cache_for_generation(
            self._prompt_cache, current_model_key, request.prompt_tokens
        )

        (uid,) = batch_generator.insert(
            [rest],
            [request.max_tokens],
            caches=[cache],
            all_tokens=[cached_prefix],
            samplers=[request.sampler],
            logits_processors=[request.logits_processors],
        )

        self._batch_results[uid] = {
            "cache_key": request.prompt_tokens[:],
            "response_queue": request.response_queue,
            "detokenizer": self.tokenizer.detokenizer,
            "top_logprobs": request.top_logprobs,
            "request_id": request.request_id,
        }

    def _generate(self) -> None:
        logger.info(
            "Distributed batched scheduler loop starting on rank %s/%s",
            self.group.rank(),
            self.group.size(),
        )
        generation_stream = prepare_mlx_lm_generation_stream(
            reason="distributed-batched-scheduler",
            distributed_group=self.group,
            use_default_stream=True,
        )
        batch_generator = BatchGenerator(
            self.model,
            max_tokens=10000000,
            completion_batch_size=self.max_seq_nums,
            prefill_batch_size=1,
            prefill_step_size=self.prefill_step_size,
            stop_tokens=[[token] for token in self.tokenizer.eos_token_ids],
            sampler=None,
            logits_processors=None,
            max_kv_size=self.max_kv_size,
            stream=generation_stream,
        )
        current_model_key = "lmstudio"

        while True:
            timeout: None | float = None if len(self._batch_results) > 0 else 0.1
            item = self._next_scheduler_item(timeout)

            if isinstance(item, DistributedSchedulerShutdownRequest):
                break
            if isinstance(item, CancelGenerationRequest):
                self._cancel_scheduler_request(batch_generator, item.request_id)
                continue
            if isinstance(item, DistributedSchedulerGenerationRequest):
                self._insert_scheduler_request(batch_generator, current_model_key, item)
                continue
            if len(self._batch_results) == 0:
                continue

            for _generation_step_index in range(SCHEDULER_GENERATION_STEPS_PER_TICK):
                prompt_responses, generation_responses = batch_generator.next()
                if not prompt_responses and not generation_responses:
                    break

                for response in prompt_responses:
                    result = self._batch_results.get(response.uid)
                    if result is None:
                        continue
                    response_queue = result.get("response_queue")
                    if response_queue is not None:
                        processed, total = response.progress
                        response_queue.put((min(processed, total), total))

                for response in generation_responses:
                    result = self._batch_results[response.uid]
                    result["cache_key"].append(response.token)
                    response_queue = result.get("response_queue")

                    if response_queue is not None:
                        detokenizer = result["detokenizer"]
                        if response.finish_reason != "stop":
                            detokenizer.add_token(response.token)
                        if response.finish_reason is not None:
                            detokenizer.finalize()

                        token_logprob = response.logprobs[response.token].item()
                        top_logprobs_list = None
                        if result["top_logprobs"] > 0:
                            sorted_indices = mx.argpartition(
                                -response.logprobs,
                                kth=result["top_logprobs"] - 1,
                            )
                            top_indices = sorted_indices[: result["top_logprobs"]]
                            top_logprobs_values = response.logprobs[top_indices]
                            top_logprobs_list = [
                                Token(
                                    id=int(token_index),
                                    text=self.tokenizer.decode(token_index),
                                    logprob=float(token_logprob_value),
                                )
                                for token_index, token_logprob_value in zip(
                                    top_indices.tolist(), top_logprobs_values.tolist()
                                )
                            ]

                        response_queue.put(
                            BatchedGenerationResponse(
                                text=detokenizer.last_segment,
                                token=response.token,
                                token_logprob=token_logprob,
                                top_logprobs=top_logprobs_list,
                                finish_reason=response.finish_reason,
                                from_draft=False,
                            )
                        )

                    if response.finish_reason is not None:
                        if response_queue is not None:
                            response_queue.put(None)
                        self._prompt_cache.insert_cache(
                            current_model_key,
                            result["cache_key"],
                            response.prompt_cache,
                        )
                        del self._batch_results[response.uid]

        self._shutdown.set()
        for entry in self._batch_results.values():
            response_queue = entry.get("response_queue")
            if response_queue is not None:
                response_queue.put(RequestCancelled("Model shutdown requested"))
