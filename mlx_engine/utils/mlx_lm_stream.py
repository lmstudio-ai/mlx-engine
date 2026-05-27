import logging
import threading
from typing import Any

import mlx.core as mx
import mlx_lm.generate as mlx_lm_generate


logger = logging.getLogger(__name__)


def _format_distributed_group(distributed_group: Any | None) -> str:
    if distributed_group is None:
        return "none"
    try:
        return f"{distributed_group.rank()}/{distributed_group.size()}"
    except Exception as caught_error:
        return f"unknown({caught_error})"


def prepare_mlx_lm_generation_stream(
    *,
    reason: str,
    request_id: str | None = None,
    distributed_group: Any | None = None,
):
    current_thread = threading.current_thread()
    thread_ident = current_thread.ident
    default_device = mx.default_device()
    previous_stream = getattr(mlx_lm_generate, "generation_stream", None)
    generation_stream = mx.new_thread_local_stream(default_device)
    mlx_lm_generate.generation_stream = generation_stream
    logger.info(
        "Prepared MLX-LM generation stream reason=%s request_id=%s rank=%s "
        "thread=%s thread_ident=%s device=%s previous_stream=%r stream=%r",
        reason,
        request_id,
        _format_distributed_group(distributed_group),
        current_thread.name,
        thread_ident,
        default_device,
        previous_stream,
        generation_stream,
    )
    return generation_stream


def log_mlx_generation_exception(
    *,
    reason: str,
    request_id: str | None = None,
    distributed_group: Any | None = None,
) -> None:
    current_thread = threading.current_thread()
    logger.exception(
        "MLX generation failed after stream preparation reason=%s request_id=%s "
        "rank=%s thread=%s thread_ident=%s stream=%r",
        reason,
        request_id,
        _format_distributed_group(distributed_group),
        current_thread.name,
        current_thread.ident,
        getattr(mlx_lm_generate, "generation_stream", None),
    )
