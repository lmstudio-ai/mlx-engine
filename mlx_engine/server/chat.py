import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
from typing import Annotated, Any, Callable, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from mlx_engine.openai_tool_calling import (
    ToolCallingPlan,
    ToolCallingValidationError,
    build_tool_calling_plan,
)


_CHAT_TEMPLATE_CONTROL_KEYS = {
    "add_generation_prompt",
    "chat_template",
    "continue_final_message",
    "conversation",
    "documents",
    "load_audio_from_video",
    "max_length",
    "messages",
    "padding",
    "processor_kwargs",
    "return_assistant_tokens_mask",
    "return_dict",
    "return_tensors",
    "tokenize",
    "tokenizer_kwargs",
    "tool_choice",
    "tools",
    "truncation",
}


class ChatRequestError(ValueError):
    """The chat request does not match the server contract."""


class _ImageDataUrl(BaseModel):
    url: str


class _TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class _InlineImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: _ImageDataUrl


_ContentPart = Annotated[
    _TextContentPart | _InlineImageContentPart,
    Field(discriminator="type"),
]
_NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
_PositiveInt = Annotated[int, Field(gt=0, strict=True)]
_NonEmptyString = Annotated[str, Field(min_length=1)]
_TopK = Annotated[int, Field(ge=-1, le=500, strict=True)]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[_ContentPart] | None = None


class _JsonSchemaDefinition(BaseModel):
    schema_: object = Field(alias="schema")


class _JsonSchemaResponseFormat(BaseModel):
    type: Literal["json_schema"]
    json_schema: _JsonSchemaDefinition


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[ChatMessage]
    stream: Literal[True]
    temperature: _NonNegativeFloat
    max_tokens: _PositiveInt | None = None
    max_completion_tokens: _PositiveInt | None = None
    stop: list[_NonEmptyString] | None = None
    top_p: _Probability | None = None
    top_k: _TopK
    min_p: _Probability | None = None
    repeat_penalty: _NonNegativeFloat | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    logit_bias: dict | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool = False
    response_format: _JsonSchemaResponseFormat | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class ChatGenerationRequest:
    prompt_tokens: list[int]
    generation_kwargs: dict[str, object]
    tool_calling_plan: ToolCallingPlan


def _validate_image_pixel_count(pixel_count: int) -> None:
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    if max_image_pixels is not None and pixel_count > max_image_pixels:
        raise ChatRequestError("Image dimensions are too large.")


def _validate_image_data(data: bytes) -> int:
    try:
        with Image.open(BytesIO(data)) as image:
            pixel_count = image.width * image.height
            _validate_image_pixel_count(pixel_count)
            image.verify()
            return pixel_count
    except Image.DecompressionBombError as error:
        raise ChatRequestError("Image dimensions are too large.") from error
    except (OSError, SyntaxError) as error:
        raise ChatRequestError("Images must contain supported image data.") from error


def _base64_image_data(url: str) -> tuple[str, int]:
    header, separator, data = url.partition(",")
    if (
        separator == ""
        or not header.startswith("data:image/")
        or not header.endswith(";base64")
    ):
        raise ChatRequestError("Images must use inline base64 data URLs.")
    if data == "":
        raise ChatRequestError("Images must contain valid base64 data.")
    try:
        image_data = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ChatRequestError("Images must contain valid base64 data.") from error
    return data, _validate_image_data(image_data)


def normalize_messages(
    messages: list[ChatMessage],
    *,
    supports_vision: bool,
) -> tuple[list[dict], list[str]]:
    normalized_messages: list[dict] = []
    images_b64: list[str] = []
    total_image_pixels = 0

    for message in messages:
        normalized_message = message.model_dump(exclude_unset=True)
        if isinstance(message.content, list):
            normalized_parts: list[dict] = []
            text_parts: list[str] = []
            for part in message.content:
                if isinstance(part, _TextContentPart):
                    text_parts.append(part.text)
                    normalized_parts.append({"type": "text", "text": part.text})
                else:
                    image_b64, image_pixels = _base64_image_data(part.image_url.url)
                    total_image_pixels += image_pixels
                    _validate_image_pixel_count(total_image_pixels)
                    images_b64.append(image_b64)
                    normalized_parts.append({"type": "image"})
            normalized_message["content"] = (
                normalized_parts if supports_vision else "".join(text_parts)
            )
        normalized_messages.append(normalized_message)

    return normalized_messages, images_b64


def _get_chat_template(model_kit: object, *, supports_vision: bool) -> Callable:
    if not supports_vision:
        return model_kit.tokenizer._tokenizer.apply_chat_template

    processor = model_kit.processor
    if getattr(processor, "chat_template", None) is not None:
        return processor.apply_chat_template
    return processor.tokenizer.apply_chat_template


def prepare_chat_generation_request(
    body: object,
    *,
    model_kit: object,
    supports_vision: bool,
    tokenize: Callable[[object, str], list[int]],
) -> ChatGenerationRequest:
    request = ChatCompletionRequest.model_validate(body)

    unsupported_controls = [
        name
        for name, requested in (
            ("max_completion_tokens", request.max_completion_tokens is not None),
            ("seed", request.seed is not None),
            ("logprobs", request.logprobs is True),
            ("top_logprobs", request.top_logprobs is not None),
            ("logit_bias", request.logit_bias is not None),
            ("presence_penalty", request.presence_penalty is not None),
            ("frequency_penalty", request.frequency_penalty is not None),
        )
        if requested
    ]
    if unsupported_controls:
        names = ", ".join(unsupported_controls)
        raise ChatRequestError(f"Unsupported generation controls: {names}.")

    has_assistant_prefill = bool(
        request.messages and request.messages[-1].role == "assistant"
    )
    if request.response_format is not None and has_assistant_prefill:
        raise ChatRequestError(
            "Structured output is not supported with assistant prefills."
        )

    normalized_messages, images_b64 = normalize_messages(
        request.messages,
        supports_vision=supports_vision,
    )
    if images_b64 and not supports_vision:
        raise ChatRequestError("The loaded model does not support images.")

    overridden_controls = _CHAT_TEMPLATE_CONTROL_KEYS.intersection(
        request.chat_template_kwargs
    )
    if overridden_controls:
        names = ", ".join(sorted(overridden_controls))
        raise ChatRequestError(
            f"chat_template_kwargs cannot override server rendering controls: {names}."
        )

    response_json_schema = _response_json_schema(request)
    try:
        tool_calling_plan = build_tool_calling_plan(
            messages=normalized_messages,
            tools=request.tools,
            tool_choice_value=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            response_json_schema=response_json_schema,
        )
    except ToolCallingValidationError as error:
        raise ChatRequestError(str(error)) from error

    template_kwargs = dict(request.chat_template_kwargs)
    if tool_calling_plan.template_tools is not None:
        template_kwargs["tools"] = tool_calling_plan.template_tools
    if tool_calling_plan.template_tool_choice is not None:
        template_kwargs["tool_choice"] = tool_calling_plan.template_tool_choice

    if has_assistant_prefill:
        template_kwargs["continue_final_message"] = True
        add_generation_prompt = False
    else:
        add_generation_prompt = True

    prompt = _get_chat_template(
        model_kit,
        supports_vision=supports_vision,
    )(
        tool_calling_plan.prompt_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )

    generation_kwargs: dict[str, object] = {
        "images_b64": images_b64,
        "temp": request.temperature,
        "top_k": request.top_k,
    }
    if tool_calling_plan.generation_json_schema is not None:
        generation_kwargs["json_schema"] = tool_calling_plan.generation_json_schema
    for name, value in (
        ("max_tokens", request.max_tokens),
        ("stop_strings", request.stop),
        ("top_p", request.top_p),
        ("min_p", request.min_p),
        ("repetition_penalty", request.repeat_penalty),
    ):
        if value is not None:
            generation_kwargs[name] = value

    return ChatGenerationRequest(
        prompt_tokens=tokenize(model_kit, prompt),
        generation_kwargs=generation_kwargs,
        tool_calling_plan=tool_calling_plan,
    )


def _response_json_schema(request: ChatCompletionRequest) -> str | None:
    if request.response_format is None:
        return None
    return json.dumps(request.response_format.json_schema.schema_)
