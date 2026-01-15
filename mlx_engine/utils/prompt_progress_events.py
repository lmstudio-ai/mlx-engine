from dataclasses import dataclass
from typing import Protocol, Union


@dataclass
class PromptProgressBeginEvent:
    cached_tokens: int
    total_prompt_tokens: int
    prefill_tokens_processed: int


@dataclass
class PromptProgressEvent:
    prefill_tokens_processed: int


class V2ProgressCallback(Protocol):
    """
    Callback for receiving prompt processing progress updates.

    Receives either a BeginEvent or ProgressEvent, plus an is_draft flag,
    and returns a bool indicating whether to continue processing (True) or cancel (False).
    """

    def __call__(
        self,
        event: Union[PromptProgressBeginEvent, PromptProgressEvent],
        is_draft: bool,
    ) -> bool: ...
