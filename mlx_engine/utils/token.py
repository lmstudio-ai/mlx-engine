from dataclasses import dataclass
from typing import Optional


@dataclass
class Token:
    """
    Base dataclass for a single generated token.
    """

    id: int
    text: str
    logprob: float
    from_draft: Optional[bool] = None
