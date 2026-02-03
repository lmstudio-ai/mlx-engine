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

    Attributes:
        text (str): The decoded text segment for this token
        token (int): The generated token ID
        token_logprob (float): Log probability of the generated token
        top_logprobs (list[Token] | None): Top token probabilities if requested
        finish_reason (str | None): Reason for completion (e.g., "stop", "length") if finished
        from_draft (bool): Whether this token came from draft model (currently always False)
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

    Attributes:
        rqueue (Queue): Response queue for streaming results back to the caller
        prompt_tokens (list[int]): Token IDs for the input prompt
        request_id (str): Unique identifier for this request
        samplers (object): Sampling function for token selection
        logits_processors (list): List of logits processors to apply
        top_logprobs (int): Number of top token probabilities to return
        max_tokens (int): Maximum number of tokens to generate
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

    Attributes:
        request_id (str): Unique identifier of the request to cancel
    """

    request_id: str
