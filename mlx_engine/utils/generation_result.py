from typing import List, Literal, NamedTuple, Optional
from mlx_engine.utils.token import Token


StopReason = Literal["eos_token", "stop_string", "user_cancelled"]


class GenerationStopCondition(NamedTuple):
    stop_reason: StopReason
    stop_string: str
    # sequence of token ids that the stop string was found in
    stop_tokens: List[int]


class GenerationResult(NamedTuple):
    text: str
    tokens: List[Token]
    top_logprobs: List[List[Token]]
    stop_condition: Optional[GenerationStopCondition]


def construct_user_cancelled_result():
    return GenerationResult(
        text="",
        tokens=[],
        top_logprobs=[],
        stop_condition=GenerationStopCondition(
            stop_reason="user_cancelled",
            stop_string="",
            stop_tokens=[],
        ),
    )
