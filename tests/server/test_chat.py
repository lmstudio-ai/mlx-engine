import json

import pytest
from pydantic import ValidationError

from mlx_engine.server.chat import (
    ChatMessage,
    ChatRequestError,
    normalize_messages,
    prepare_chat_generation_request,
)


class _FakeRenderer:
    def __init__(self):
        self.chat_template = "model template"
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered prompt"


class _FakeTokenizerWrapper:
    def __init__(self, renderer):
        self._tokenizer = renderer


class _FakeTextModelKit:
    def __init__(self, renderer):
        self.tokenizer = _FakeTokenizerWrapper(renderer)


class _FakeVisionModelKit:
    def __init__(self, renderer):
        self.processor = renderer


def _base_request(**overrides):
    request = {
        "model": "ignored-single-model-id",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
        "max_tokens": 100,
        "stop": ["END"],
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
    }
    request.update(overrides)
    return request


def test_prepare_text_request_uses_only_supported_generation_settings():
    renderer = _FakeRenderer()
    model_kit = _FakeTextModelKit(renderer)
    tokenization_calls = []

    request = prepare_chat_generation_request(
        _base_request(chat_template_kwargs={"reasoning_effort": "medium"}),
        model_kit=model_kit,
        supports_vision=False,
        tokenize=lambda received_model_kit, prompt: tokenization_calls.append(
            (received_model_kit, prompt)
        )
        or [1, 2, 3],
    )

    assert request.prompt_tokens == [1, 2, 3]
    assert request.generation_kwargs == {
        "images_b64": [],
        "temp": 0.7,
        "max_tokens": 100,
        "stop_strings": ["END"],
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    }
    assert tokenization_calls == [(model_kit, "rendered prompt")]

    messages, template_kwargs = renderer.calls[0]
    assert messages == [{"role": "user", "content": "Hello"}]
    assert template_kwargs["tokenize"] is False
    assert template_kwargs["add_generation_prompt"] is True
    assert "continue_final_message" not in template_kwargs
    assert template_kwargs["reasoning_effort"] == "medium"


def test_tools_are_rejected():
    renderer = _FakeRenderer()

    with pytest.raises(ChatRequestError, match="Tools are not supported yet"):
        prepare_chat_generation_request(
            _base_request(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "search"},
                    }
                ]
            ),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


def test_json_schema_is_forwarded_to_generation():
    renderer = _FakeRenderer()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    request = prepare_chat_generation_request(
        _base_request(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        ),
        model_kit=_FakeTextModelKit(renderer),
        supports_vision=False,
        tokenize=lambda _model_kit, _prompt: [],
    )

    assert json.loads(request.generation_kwargs["json_schema"]) == schema


def test_unsupported_response_format_is_rejected():
    with pytest.raises(ValidationError, match="json_schema"):
        prepare_chat_generation_request(
            _base_request(response_format={"type": "json_object"}),
            model_kit=_FakeTextModelKit(_FakeRenderer()),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )


def test_structured_output_with_assistant_prefill_is_rejected():
    renderer = _FakeRenderer()

    with pytest.raises(
        ChatRequestError,
        match="Structured output is not supported with assistant prefills",
    ):
        prepare_chat_generation_request(
            _base_request(
                messages=[
                    {"role": "user", "content": "Respond with JSON"},
                    {"role": "assistant", "content": '{"answer":'},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
            ),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


def test_final_assistant_message_is_rendered_as_a_prefill():
    renderer = _FakeRenderer()

    prepare_chat_generation_request(
        _base_request(
            messages=[
                {"role": "user", "content": "Respond with JSON"},
                {"role": "assistant", "content": '{"answer":'},
            ]
        ),
        model_kit=_FakeTextModelKit(renderer),
        supports_vision=False,
        tokenize=lambda _model_kit, _prompt: [],
    )

    messages, template_kwargs = renderer.calls[0]
    assert messages[-1] == {"role": "assistant", "content": '{"answer":'}
    assert template_kwargs["add_generation_prompt"] is False
    assert template_kwargs["continue_final_message"] is True


@pytest.mark.parametrize(
    "control_name",
    [
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "tokenize",
        "tools",
    ],
)
def test_chat_template_kwargs_cannot_override_server_controls(control_name):
    renderer = _FakeRenderer()

    with pytest.raises(ChatRequestError, match="server rendering controls"):
        prepare_chat_generation_request(
            _base_request(chat_template_kwargs={control_name: "override"}),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


def test_normalize_images_preserves_user_and_tool_result_order():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,first-image",
                        "detail": "auto",
                    },
                },
                {"type": "text", "text": "Second"},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "previous reasoning",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "view", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [
                {"type": "text", "text": "Result"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,second-image",
                        "detail": "auto",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,first-image",
                        "detail": "auto",
                    },
                },
            ],
        },
    ]

    normalized, images_b64 = normalize_messages(
        [ChatMessage.model_validate(message) for message in messages]
    )

    assert images_b64 == ["first-image", "second-image", "first-image"]
    assert normalized[0]["content"] == [
        {"type": "text", "text": "First"},
        {"type": "image"},
        {"type": "text", "text": "Second"},
    ]
    assert normalized[1] == messages[1]
    assert normalized[2]["tool_call_id"] == "call-1"
    assert normalized[2]["content"] == [
        {"type": "text", "text": "Result"},
        {"type": "image"},
        {"type": "image"},
    ]


def test_prepare_vision_request_forwards_base64_to_generation_boundary():
    renderer = _FakeRenderer()
    model_kit = _FakeVisionModelKit(renderer)
    request = prepare_chat_generation_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,image-payload",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]
        ),
        model_kit=model_kit,
        supports_vision=True,
        tokenize=lambda _model_kit, _prompt: [7, 8],
    )

    assert request.generation_kwargs["images_b64"] == ["image-payload"]
    assert renderer.calls[0][0][0]["content"] == [
        {"type": "text", "text": "Describe this"},
        {"type": "image"},
    ]


def test_vision_request_uses_processor_tokenizer_when_processor_template_is_missing():
    tokenizer_renderer = _FakeRenderer()
    tokenizer_renderer.chat_template = "model template"

    class ProcessorWithoutTemplate:
        chat_template = None
        tokenizer = tokenizer_renderer

        def apply_chat_template(self, _messages, **_kwargs):
            raise AssertionError("processor without template must not render")

    model_kit = _FakeVisionModelKit(ProcessorWithoutTemplate())
    request = prepare_chat_generation_request(
        _base_request(),
        model_kit=model_kit,
        supports_vision=True,
        tokenize=lambda _model_kit, _prompt: [7, 8],
    )

    assert request.prompt_tokens == [7, 8]
    assert len(tokenizer_renderer.calls) == 1


def test_non_base64_image_url_is_rejected():
    with pytest.raises(ChatRequestError, match="inline base64"):
        normalize_messages(
            [
                ChatMessage.model_validate(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            }
                        ],
                    }
                )
            ]
        )


def test_text_model_rejects_image_request():
    renderer = _FakeRenderer()
    with pytest.raises(ChatRequestError, match="does not support images"):
        prepare_chat_generation_request(
            _base_request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,image-payload"
                                },
                            }
                        ],
                    }
                ]
            ),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )
