from dataclasses import dataclass
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    normalized_messages, images_b64 = normalize_messages(request.messages)
    if images_b64 and not supports_vision:
        raise ChatRequestError("The loaded model does not support images.")

    template_kwargs = dict(request.chat_template_kwargs)
    if request.tools:
        template_kwargs["tools"] = request.tools
    prompt = _get_chat_template(
        model_kit,
        supports_vision=supports_vision,
    )(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    generation_kwargs: dict[str, object] = {
        "images_b64": images_b64,
        "temp": request.temperature,
        "top_k": request.top_k,
    }
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
