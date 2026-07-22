from dataclasses import dataclass
import re
from typing import Callable


_INLINE_IMAGE_PATTERN = re.compile(r"^data:image/[^;,]+;base64,(.*)$", re.DOTALL)


class ChatRequestError(ValueError):
    """The chat request does not match the server contract."""


@dataclass(frozen=True)
class ChatGenerationRequest:
    prompt: str
    prompt_tokens: list[int]
    images_b64: list[str]
    temperature: float
    max_tokens: int | None
    stop_strings: list[str] | None
    top_p: float | None
    top_k: int
    min_p: float | None
    repetition_penalty: float | None


def _normalize_content(
    content: object,
    *,
    role: str,
    images_b64: list[str],
) -> object:
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        raise ChatRequestError(
            f"Message content for role '{role}' must be text or parts."
        )

    normalized_parts: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            raise ChatRequestError("Message content parts must be objects.")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ChatRequestError("Text content parts must contain text.")
            normalized_parts.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            if role not in ("user", "tool"):
                raise ChatRequestError(
                    f"Images are not supported in '{role}' messages."
                )
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                raise ChatRequestError("Image content parts must contain image_url.")
            url = image_url.get("url")
            if not isinstance(url, str):
                raise ChatRequestError("Image URLs must be strings.")
            match = _INLINE_IMAGE_PATTERN.fullmatch(url)
            if match is None:
                raise ChatRequestError("Images must use inline base64 data URLs.")
            images_b64.append(match.group(1))
            normalized_parts.append({"type": "image"})
            continue
        raise ChatRequestError(f"Unsupported message content part type: {part_type!r}.")
    return normalized_parts


def normalize_messages(messages: object) -> tuple[list[dict], list[str]]:
    if not isinstance(messages, list):
        raise ChatRequestError("messages must be an array.")

    normalized_messages: list[dict] = []
    images_b64: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ChatRequestError("messages must contain objects.")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ChatRequestError(f"Unsupported message role: {role!r}.")

        normalized_message = dict(message)
        if "content" in normalized_message:
            normalized_message["content"] = _normalize_content(
                normalized_message["content"],
                role=role,
                images_b64=images_b64,
            )
        normalized_messages.append(normalized_message)

    return normalized_messages, images_b64


def _get_number(body: dict, field_name: str, *, required: bool) -> int | float | None:
    value = body.get(field_name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatRequestError(f"{field_name} must be a number.")
    return value


def _get_chat_template(model_kit: object, *, supports_vision: bool) -> Callable:
    tokenizer = getattr(model_kit, "tokenizer", None)
    if supports_vision:
        processor = getattr(model_kit, "processor", None)
        candidates = [
            processor,
            getattr(processor, "tokenizer", None),
            tokenizer,
            getattr(tokenizer, "_tokenizer", None),
        ]
    else:
        candidates = [getattr(tokenizer, "_tokenizer", None), tokenizer]

    for renderer in candidates:
        apply_chat_template = getattr(renderer, "apply_chat_template", None)
        if (
            callable(apply_chat_template)
            and getattr(renderer, "chat_template", None) is not None
        ):
            return apply_chat_template
    for renderer in candidates:
        apply_chat_template = getattr(renderer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return apply_chat_template
    raise ChatRequestError("The loaded model does not provide a chat template.")


def prepare_chat_generation_request(
    body: object,
    *,
    model_kit: object,
    supports_vision: bool,
    tokenize: Callable[[object, str], list[int]],
) -> ChatGenerationRequest:
    if not isinstance(body, dict):
        raise ChatRequestError("The request body must be an object.")
    if body.get("stream") is not True:
        raise ChatRequestError("Streaming generation requires stream=true.")

    normalized_messages, images_b64 = normalize_messages(body.get("messages"))
    if len(images_b64) > 0 and not supports_vision:
        raise ChatRequestError("The loaded model does not support images.")

    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ChatRequestError("tools must be an array.")
    chat_template_kwargs = body.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    if not isinstance(chat_template_kwargs, dict):
        raise ChatRequestError("chat_template_kwargs must be an object.")

    template_kwargs = dict(chat_template_kwargs)
    if tools is not None and len(tools) > 0:
        template_kwargs["tools"] = tools
    apply_chat_template = _get_chat_template(
        model_kit,
        supports_vision=supports_vision,
    )
    prompt = apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not isinstance(prompt, str):
        raise ChatRequestError("The model chat template did not return text.")

    stop_strings = body.get("stop")
    if stop_strings is not None and (
        not isinstance(stop_strings, list)
        or any(not isinstance(stop_string, str) for stop_string in stop_strings)
    ):
        raise ChatRequestError("stop must be an array of strings.")

    max_tokens = _get_number(body, "max_tokens", required=False)
    if max_tokens is not None and not isinstance(max_tokens, int):
        raise ChatRequestError("max_tokens must be an integer.")
    top_k = _get_number(body, "top_k", required=True)
    if not isinstance(top_k, int):
        raise ChatRequestError("top_k must be an integer.")

    return ChatGenerationRequest(
        prompt=prompt,
        prompt_tokens=tokenize(model_kit, prompt),
        images_b64=images_b64,
        temperature=float(_get_number(body, "temperature", required=True)),
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        top_p=(
            None
            if (top_p := _get_number(body, "top_p", required=False)) is None
            else float(top_p)
        ),
        top_k=top_k,
        min_p=(
            None
            if (min_p := _get_number(body, "min_p", required=False)) is None
            else float(min_p)
        ),
        repetition_penalty=(
            None
            if (
                repetition_penalty := _get_number(
                    body, "repeat_penalty", required=False
                )
            )
            is None
            else float(repetition_penalty)
        ),
    )
