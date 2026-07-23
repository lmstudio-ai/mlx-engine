from dataclasses import dataclass
import json
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


_CHAT_TEMPLATE_CONTROL_KEYS = {
    "add_generation_prompt",
    "chat_template",
    "continue_final_message",
    "tokenize",
    "tools",
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
    temperature: float
    max_tokens: int | None = None
    stop: list[str] | None = None
    top_p: float | None = None
    top_k: int
    min_p: float | None = None
    repeat_penalty: float | None = None
    tools: list[dict] | None = None
    response_format: _JsonSchemaResponseFormat | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class ChatGenerationRequest:
    prompt_tokens: list[int]
    generation_kwargs: dict[str, object]


def _base64_image_data(url: str) -> str:
    header, separator, data = url.partition(",")
    if (
        separator == ""
        or not header.startswith("data:image/")
        or not header.endswith(";base64")
    ):
        raise ChatRequestError("Images must use inline base64 data URLs.")
    return data


def normalize_messages(messages: list[ChatMessage]) -> tuple[list[dict], list[str]]:
    normalized_messages: list[dict] = []
    images_b64: list[str] = []

    for message in messages:
        normalized_message = message.model_dump(exclude_unset=True)
        if isinstance(message.content, list):
            normalized_parts: list[dict] = []
            for part in message.content:
                if isinstance(part, _TextContentPart):
                    normalized_parts.append({"type": "text", "text": part.text})
                else:
                    images_b64.append(_base64_image_data(part.image_url.url))
                    normalized_parts.append({"type": "image"})
            normalized_message["content"] = normalized_parts
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
    if request.tools:
        raise ChatRequestError("Tools are not supported yet.")

    normalized_messages, images_b64 = normalize_messages(request.messages)
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

    template_kwargs = dict(request.chat_template_kwargs)
    if request.messages and request.messages[-1].role == "assistant":
        template_kwargs["continue_final_message"] = True
        add_generation_prompt = False
    else:
        add_generation_prompt = True

    prompt = _get_chat_template(
        model_kit,
        supports_vision=supports_vision,
    )(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )

    generation_kwargs: dict[str, object] = {
        "images_b64": images_b64,
        "temp": request.temperature,
        "top_k": request.top_k,
    }
    if request.response_format is not None:
        generation_kwargs["json_schema"] = json.dumps(
            request.response_format.json_schema.schema_
        )
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
    )
