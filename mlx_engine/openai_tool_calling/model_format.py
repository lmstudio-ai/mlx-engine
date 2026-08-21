from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Literal

from mlx_engine.openai_tool_calling.model_parsers import (
    Gemma4ToolCallParser,
    ModelToolCallParser,
    MuseGlimmerToolCallParser,
    Qwen35ToolCallParser,
    parse_gemma4_arguments_object,
)
from mlx_engine.openai_tool_calling.model_parsers.upstream import (
    FallbackToolCallParser,
    UpstreamToolParserAdapter,
)
from mlx_engine.openai_tool_calling.models import (
    FunctionToolSpec,
    JsonObject,
    ToolCallIdFactory,
)
from mlx_engine.tool_protocols import (
    GEMMA4_TOOL_CALL_END,
    GEMMA4_TOOL_CALL_START,
    MUSE_GLIMMER_ATEM_END,
    MUSE_GLIMMER_ATEM_START,
    QWEN35_TOOL_CALL_END,
    QWEN35_TOOL_CALL_START,
)

ModelToolCallFormat = Literal["auto", "qwen35", "gemma4", "muse_glimmer"]
SelectedModelToolCallFormat = Literal["qwen35", "gemma4", "muse_glimmer"]


@dataclass(frozen=True)
class ModelToolCallFormatSpec:
    name: SelectedModelToolCallFormat
    start_marker: str
    end_marker: str
    parser: ModelToolCallParser
    parser_type_aliases: tuple[str, ...] = ()
    model_type_values: tuple[str, ...] = ()
    model_type_prefixes: tuple[str, ...] = ()
    chat_template_markers: tuple[tuple[str, ...], ...] = ()
    upstream_parser_modules: tuple[str, ...] = ()

    def matches_parser_type(self, parser_type: str | None) -> bool:
        return _normalize_key(parser_type) in self.parser_type_aliases

    def matches_model_type(self, model_type: str | None) -> bool:
        normalized_model_type = str(model_type or "")
        return normalized_model_type in self.model_type_values or any(
            normalized_model_type.startswith(prefix)
            for prefix in self.model_type_prefixes
        )

    def matches_chat_template(self, chat_template: object) -> bool:
        if not isinstance(chat_template, str):
            return False
        return any(
            all(marker in chat_template for marker in markers)
            for markers in self.chat_template_markers
        )


def parse_model_format_tool_calls(
    model_output: str,
    tool_specs: list[FunctionToolSpec],
    *,
    id_factory: ToolCallIdFactory | None = None,
    model_format: ModelToolCallFormat = "auto",
) -> list[JsonObject]:
    """Parse supported model-format MLX tool-call text into OpenAI tool calls."""
    selected_format = select_model_tool_call_format(model_output, model_format)
    if selected_format is None:
        return []
    return get_model_tool_call_format_spec(selected_format).parser.parse(
        model_output,
        tool_specs,
        id_factory=id_factory,
    )


def select_model_tool_call_format(
    model_output: str,
    model_format: ModelToolCallFormat,
) -> SelectedModelToolCallFormat | None:
    if model_format != "auto":
        return model_format
    marker_positions = [
        (position, spec.name)
        for spec in _FORMAT_SPECS
        if (position := model_output.find(spec.start_marker)) >= 0
    ]
    if len(marker_positions) == 0:
        return None
    return min(marker_positions)[1]


def resolve_model_tool_call_format(
    model_kit: object,
    *,
    supports_vision: bool,
) -> ModelToolCallFormat:
    for parser_type in _runtime_parser_types(model_kit, supports_vision=supports_vision):
        if model_format := model_tool_call_format_from_parser_type(parser_type):
            return model_format

    for chat_template in _runtime_chat_templates(model_kit, supports_vision=supports_vision):
        if model_format := model_tool_call_format_from_chat_template(chat_template):
            return model_format

    return model_tool_call_format_from_model_type(getattr(model_kit, "model_type", None))


def model_tool_call_format_from_parser_type(
    parser_type: str | None,
) -> SelectedModelToolCallFormat | None:
    for spec in _FORMAT_SPECS:
        if spec.matches_parser_type(parser_type):
            return spec.name
    return None


def model_tool_call_format_from_chat_template(
    chat_template: object,
) -> SelectedModelToolCallFormat | None:
    for spec in _FORMAT_SPECS:
        if spec.matches_chat_template(chat_template):
            return spec.name
    return None


def model_tool_call_format_from_model_type(
    model_type: str | None,
) -> ModelToolCallFormat:
    for spec in _FORMAT_SPECS:
        if spec.matches_model_type(model_type):
            return spec.name
    return "auto"


def runtime_matches_tool_call_format(
    *,
    model_type: str | None,
    tokenizer: object,
    model_format: SelectedModelToolCallFormat,
) -> bool:
    spec = get_model_tool_call_format_spec(model_format)
    return (
        spec.matches_model_type(model_type)
        or any(
            spec.matches_parser_type(parser_type)
            for parser_type in _tokenizer_parser_types(tokenizer)
        )
        or any(
            spec.matches_chat_template(template)
            for template in _tokenizer_chat_templates(tokenizer)
        )
    )


def get_model_tool_call_format_spec(
    model_format: SelectedModelToolCallFormat,
) -> ModelToolCallFormatSpec:
    return _FORMAT_SPEC_BY_NAME[model_format]


def _runtime_parser_types(
    model_kit: object,
    *,
    supports_vision: bool,
) -> list[str]:
    parser_types = []
    if tokenizer := getattr(model_kit, "tokenizer", None):
        parser_types.extend(_tokenizer_parser_types(tokenizer))
    if supports_vision and (processor := getattr(model_kit, "processor", None)):
        parser_types.extend(_processor_parser_types(processor))
        if processor_tokenizer := getattr(processor, "tokenizer", None):
            parser_types.extend(_tokenizer_parser_types(processor_tokenizer))
    return _dedupe(parser_types)


def _tokenizer_parser_types(tokenizer: object) -> list[str]:
    parser_types: list[str] = []
    for candidate in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        init_kwargs = getattr(candidate, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            parser_type = init_kwargs.get("tool_parser_type")
            if isinstance(parser_type, str):
                parser_types.append(parser_type)
    for attribute in ("tool_parser", "_tool_parser"):
        parser = getattr(tokenizer, attribute, None)
        module_name = getattr(parser, "__module__", "")
        if module_name:
            parser_types.append(module_name.rsplit(".", 1)[-1])
    return parser_types


def _processor_parser_types(processor: object) -> list[str]:
    parser_types: list[str] = []
    init_kwargs = getattr(processor, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        parser_type = init_kwargs.get("tool_parser_type")
        if isinstance(parser_type, str):
            parser_types.append(parser_type)
    inferred = _infer_mlx_vlm_tool_parser_type(processor)
    if inferred is not None:
        parser_types.append(inferred)
    return parser_types


def _runtime_chat_templates(
    model_kit: object,
    *,
    supports_vision: bool,
) -> list[object]:
    chat_templates: list[object] = []
    if tokenizer := getattr(model_kit, "tokenizer", None):
        chat_templates.extend(_tokenizer_chat_templates(tokenizer))
    if supports_vision and (processor := getattr(model_kit, "processor", None)):
        chat_templates.append(getattr(processor, "chat_template", None))
        if processor_tokenizer := getattr(processor, "tokenizer", None):
            chat_templates.extend(_tokenizer_chat_templates(processor_tokenizer))
    return [template for template in chat_templates if template is not None]


def _tokenizer_chat_templates(tokenizer: object) -> list[object]:
    return [
        getattr(tokenizer, "chat_template", None),
        getattr(getattr(tokenizer, "_tokenizer", None), "chat_template", None),
    ]


def _infer_mlx_vlm_tool_parser_type(processor: object) -> str | None:
    try:
        tool_parsers = importlib.import_module("mlx_vlm.tool_parsers")
    except Exception:
        return None
    infer = getattr(tool_parsers, "_infer_tool_parser_from_processor", None)
    if not callable(infer):
        return None
    try:
        inferred = infer(processor)
    except Exception:
        return None
    return inferred if isinstance(inferred, str) else None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_key(value: str | None) -> str:
    return str(value or "").replace("-", "_").rsplit(".", 1)[-1]


def _with_upstream_fallback(spec: ModelToolCallFormatSpec) -> ModelToolCallFormatSpec:
    if len(spec.upstream_parser_modules) == 0:
        return spec
    return ModelToolCallFormatSpec(
        name=spec.name,
        start_marker=spec.start_marker,
        end_marker=spec.end_marker,
        parser=FallbackToolCallParser(
            primary=spec.parser,
            fallback=UpstreamToolParserAdapter(
                module_names=spec.upstream_parser_modules,
                start_marker=spec.start_marker,
                end_marker=spec.end_marker,
            ),
        ),
        parser_type_aliases=spec.parser_type_aliases,
        model_type_values=spec.model_type_values,
        model_type_prefixes=spec.model_type_prefixes,
        chat_template_markers=spec.chat_template_markers,
        upstream_parser_modules=spec.upstream_parser_modules,
    )


_FORMAT_SPECS: tuple[ModelToolCallFormatSpec, ...] = tuple(
    _with_upstream_fallback(spec)
    for spec in (
        ModelToolCallFormatSpec(
            name="qwen35",
            start_marker=QWEN35_TOOL_CALL_START,
            end_marker=QWEN35_TOOL_CALL_END,
            parser=Qwen35ToolCallParser(),
            parser_type_aliases=("qwen35", "qwen3_5", "qwen3_coder", "qwen3_xml"),
            model_type_prefixes=("qwen3_5",),
            chat_template_markers=(("<tool_call>", "<function="),),
            upstream_parser_modules=("mlx_lm.tool_parsers.qwen3_coder",),
        ),
        ModelToolCallFormatSpec(
            name="gemma4",
            start_marker=GEMMA4_TOOL_CALL_START,
            end_marker=GEMMA4_TOOL_CALL_END,
            parser=Gemma4ToolCallParser(),
            parser_type_aliases=("gemma4",),
            model_type_prefixes=("gemma4",),
            chat_template_markers=((GEMMA4_TOOL_CALL_START, GEMMA4_TOOL_CALL_END),),
            upstream_parser_modules=(
                "mlx_lm.tool_parsers.gemma4",
                "mlx_vlm.tool_parsers.gemma4",
            ),
        ),
        ModelToolCallFormatSpec(
            name="muse_glimmer",
            start_marker=MUSE_GLIMMER_ATEM_START,
            end_marker=MUSE_GLIMMER_ATEM_END,
            parser=MuseGlimmerToolCallParser(),
            parser_type_aliases=("atem", "muse_glimmer"),
            model_type_values=("muse_glimmer",),
            chat_template_markers=(
                ("atem:function_calls", "<atem:invoke"),
                (MUSE_GLIMMER_ATEM_START, "<atem:invoke"),
            ),
            upstream_parser_modules=("mlx_vlm.tool_parsers.atem",),
        ),
    )
)
_FORMAT_SPEC_BY_NAME = {spec.name: spec for spec in _FORMAT_SPECS}


__all__ = [
    "ModelToolCallFormat",
    "ModelToolCallFormatSpec",
    "SelectedModelToolCallFormat",
    "get_model_tool_call_format_spec",
    "model_tool_call_format_from_chat_template",
    "model_tool_call_format_from_model_type",
    "model_tool_call_format_from_parser_type",
    "parse_gemma4_arguments_object",
    "parse_model_format_tool_calls",
    "resolve_model_tool_call_format",
    "runtime_matches_tool_call_format",
    "select_model_tool_call_format",
]
