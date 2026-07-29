import json

import pytest
from PIL import Image
from pydantic import ValidationError
from transformers.utils.chat_template_utils import render_jinja_template

from mlx_engine.server.chat import (
    ChatMessage,
    ChatRequestError,
    normalize_messages,
    prepare_chat_generation_request,
)


_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)
_BLUE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAI"
    "icLsAAAAAElFTkSuQmCC"
)
# A 10,000 x 10,000 PNG header with no allocated pixel data.
_DECOMPRESSION_BOMB_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAJxAAACcQCAIAAAA1LPVwAAAAAElFTkSuQmCC"
)
_TRUNCATED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQ="


class _FakeRenderer:
    def __init__(self):
        self.chat_template = "model template"
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered prompt"


class _TransformersTextRenderer:
    chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] | trim }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )

    def apply_chat_template(self, messages, **kwargs):
        rendered, _generation_indices = render_jinja_template(
            conversations=[messages],
            chat_template=self.chat_template,
            **kwargs,
        )
        return rendered[0]


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


@pytest.mark.parametrize(
    ("content", "expected_prompt"),
    [
        (
            [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
            "FirstSecondassistant",
        ),
        ([], "assistant"),
    ],
)
def test_text_content_parts_are_strings_before_transformers_template(
    content,
    expected_prompt,
):
    rendered_prompts = []

    prepare_chat_generation_request(
        _base_request(messages=[{"role": "user", "content": content}]),
        model_kit=_FakeTextModelKit(_TransformersTextRenderer()),
        supports_vision=False,
        tokenize=lambda _model_kit, prompt: rendered_prompts.append(prompt) or [],
    )

    assert rendered_prompts == [expected_prompt]


def test_supported_generation_boundaries_and_unknown_future_fields_are_accepted():
    renderer = _FakeRenderer()

    request = prepare_chat_generation_request(
        _base_request(
            temperature=0,
            max_tokens=1,
            top_p=1,
            top_k=-1,
            min_p=0,
            repeat_penalty=0,
            logprobs=False,
            future_sampling_control={"enabled": True},
        ),
        model_kit=_FakeTextModelKit(renderer),
        supports_vision=False,
        tokenize=lambda _model_kit, _prompt: [],
    )

    assert request.generation_kwargs == {
        "images_b64": [],
        "temp": 0,
        "max_tokens": 1,
        "stop_strings": ["END"],
        "top_p": 1,
        "top_k": -1,
        "min_p": 0,
        "repetition_penalty": 0,
    }


@pytest.mark.parametrize(
    ("overrides", "control_name"),
    [
        ({"max_completion_tokens": 1}, "max_completion_tokens"),
        ({"seed": 0}, "seed"),
        ({"logprobs": True}, "logprobs"),
        ({"top_logprobs": 5}, "top_logprobs"),
        ({"logit_bias": {"1": 1}}, "logit_bias"),
        ({"presence_penalty": 0}, "presence_penalty"),
        ({"frequency_penalty": 0}, "frequency_penalty"),
    ],
)
def test_unsupported_generation_controls_are_rejected(overrides, control_name):
    renderer = _FakeRenderer()

    with pytest.raises(ChatRequestError, match=control_name):
        prepare_chat_generation_request(
            _base_request(**overrides),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": 1.5},
        {"stop": [""]},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"top_k": -2},
        {"top_k": 501},
        {"top_k": 1.5},
        {"min_p": -0.1},
        {"min_p": 1.1},
        {"min_p": float("nan")},
        {"repeat_penalty": -0.1},
        {"repeat_penalty": float("nan")},
    ],
)
def test_invalid_generation_settings_are_rejected_before_rendering(overrides):
    renderer = _FakeRenderer()

    with pytest.raises(ValidationError):
        prepare_chat_generation_request(
            _base_request(**overrides),
            model_kit=_FakeTextModelKit(renderer),
            supports_vision=False,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


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
        "tools",
        "truncation",
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
                        "url": f"data:image/png;base64,{_RED_PNG_B64}",
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
                        "url": f"data:image/png;base64,{_BLUE_PNG_B64}",
                        "detail": "auto",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_RED_PNG_B64}",
                        "detail": "auto",
                    },
                },
            ],
        },
    ]

    normalized, images_b64 = normalize_messages(
        [ChatMessage.model_validate(message) for message in messages],
        supports_vision=True,
    )

    assert images_b64 == [
        _RED_PNG_B64,
        _BLUE_PNG_B64,
        _RED_PNG_B64,
    ]
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
                                "url": f"data:image/png;base64,{_RED_PNG_B64}",
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

    assert request.generation_kwargs["images_b64"] == [_RED_PNG_B64]
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
            ],
            supports_vision=True,
        )


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(
    ("url", "error_message"),
    [
        ("data:image/jpeg;base64,", "valid base64 data"),
        ("data:image/jpeg;base64,not-valid-base64!", "valid base64 data"),
        ("data:image/png;base64,bm90IGFuIGltYWdl", "supported image data"),
        (
            f"data:image/png;base64,{_TRUNCATED_PNG_B64}",
            "supported image data",
        ),
        (
            f"data:image/png;base64,{_DECOMPRESSION_BOMB_PNG_B64}",
            "Image dimensions are too large",
        ),
    ],
)
def test_invalid_image_data_is_rejected_before_rendering(url, error_message):
    renderer = _FakeRenderer()

    with pytest.raises(ChatRequestError, match=error_message):
        prepare_chat_generation_request(
            _base_request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            }
                        ],
                    }
                ]
            ),
            model_kit=_FakeVisionModelKit(renderer),
            supports_vision=True,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


def test_aggregate_image_pixels_are_bounded_before_rendering(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2)
    renderer = _FakeRenderer()
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"},
    }

    with pytest.raises(ChatRequestError, match="Image dimensions are too large"):
        prepare_chat_generation_request(
            _base_request(
                messages=[
                    {
                        "role": "user",
                        "content": [image_part, image_part, image_part],
                    }
                ]
            ),
            model_kit=_FakeVisionModelKit(renderer),
            supports_vision=True,
            tokenize=lambda _model_kit, _prompt: [],
        )

    assert renderer.calls == []


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
                                    "url": f"data:image/png;base64,{_RED_PNG_B64}"
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
