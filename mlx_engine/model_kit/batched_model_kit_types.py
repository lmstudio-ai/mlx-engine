"""
Data types and structures for batched model kit operations.

This module contains the request/response types and exceptions used for
communicating with the batched generation backend.
"""

from dataclasses import dataclass
from queue import Queue
from mlx_engine.utils.token import Token


@dataclass
class BatchedGenerationResponse:
    """
    Response object for batched generation containing a single generated token.
    """

    text: str
    token: int
    token_logprob: float
    top_logprobs: list[Token] | None
    finish_reason: str | None
    from_draft: bool = False


class RequestCancelled(Exception):
    """
    Exception raised when a generation request is successfully cancelled.

    This exception is yielded to the request's response queue when the request
    is cancelled either by user request or during shutdown.
    """


@dataclass
class GenerationRequest:
    """
    Internal request object for queuing generation requests.
    """

    rqueue: Queue
    prompt_tokens: list[int]
    request_id: str
    samplers: object
    logits_processors: list
    top_logprobs: int
    max_tokens: int


@dataclass
class CancelGenerationRequest:
    """
    Internal request object for cancelling a generation request.
    """

    request_id: str
