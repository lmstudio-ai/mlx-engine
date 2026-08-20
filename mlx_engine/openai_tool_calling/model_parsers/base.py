from __future__ import annotations

import json
import math
from typing import Any, Protocol

from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
)


class ModelToolCallParser(Protocol):
    start_marker: str

    def parse(
        self,
        model_output: str,
        tool_specs: list[FunctionToolSpec],
        *,
        id_factory: ToolCallIdFactory | None = None,
    ) -> list[JsonObject]: ...


def parse_json_argument_value(value: str) -> Any:
    stripped_value = value.strip()
    if stripped_value == "":
        return ""
    try:
        return json.loads(
            stripped_value,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except ValueError:
        return stripped_value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Unsupported JSON constant: {value}")


def _parse_json_float(value: str) -> float | str:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else value
