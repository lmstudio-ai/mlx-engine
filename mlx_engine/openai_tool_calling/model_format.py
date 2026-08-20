from __future__ import annotations

from typing import Literal

from mlx_engine.openai_tool_calling.model_parsers import (
    Gemma4ToolCallParser,
    ModelToolCallParser,
    MuseGlimmerToolCallParser,
    Qwen35ToolCallParser,
    parse_gemma4_arguments_object,
)
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
)

ModelToolCallFormat = Literal["auto", "qwen35", "gemma4", "muse_glimmer"]
_SelectedModelToolCallFormat = Literal["qwen35", "gemma4", "muse_glimmer"]
_PARSERS: dict[_SelectedModelToolCallFormat, ModelToolCallParser] = {
    "qwen35": Qwen35ToolCallParser(),
    "gemma4": Gemma4ToolCallParser(),
    "muse_glimmer": MuseGlimmerToolCallParser(),
}
_MODEL_FORMAT_MARKERS: tuple[tuple[_SelectedModelToolCallFormat, str], ...] = tuple(
    (model_format, parser.start_marker) for model_format, parser in _PARSERS.items()
)


def parse_model_format_tool_calls(
    model_output: str,
    tool_specs: list[FunctionToolSpec],
    *,
    id_factory: ToolCallIdFactory | None = None,
    model_format: ModelToolCallFormat = "auto",
) -> list[JsonObject]:
    """Parse supported model-format MLX tool-call text into OpenAI tool calls."""
    selected_format = _select_model_tool_call_format(model_output, model_format)
    if selected_format is None:
        return []
    return _PARSERS[selected_format].parse(
        model_output,
        tool_specs,
        id_factory=id_factory,
    )


def _select_model_tool_call_format(
    model_output: str,
    model_format: ModelToolCallFormat,
) -> _SelectedModelToolCallFormat | None:
    if model_format != "auto":
        return model_format
    marker_positions = [
        (position, marker_format)
        for marker_format, marker in _MODEL_FORMAT_MARKERS
        if (position := model_output.find(marker)) >= 0
    ]
    if len(marker_positions) == 0:
        return None
    return min(marker_positions)[1]


__all__ = [
    "ModelToolCallFormat",
    "parse_gemma4_arguments_object",
    "parse_model_format_tool_calls",
]
