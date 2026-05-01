import mlx.core as mx

from mlx_engine.model_kit.batched_vision import prompt_inputs as prompt_inputs_module
from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptImageSpan
from mlx_engine.model_kit.batched_vision.prompt_inputs import (
    PreparedPrompt,
    build_cached_prompt_kwargs,
    build_prompt_kwargs,
    prepare_prompt_inputs,
)


class _FakeTokenizer:
    def __init__(self):
        self.decoded = []

    def decode(self, tokens):
        self.decoded.append(list(tokens))
        return "decoded prompt"

    def tokenize(self, prompt):
        return [prompt]

    def convert_tokens_to_ids(self, tokens):
        return [101 for _ in tokens]


class _FakeImage:
    mode = "RGB"
    size = (1, 1)

    def __init__(self, byte: bytes):
        self._byte = byte

    def tobytes(self):
        return self._byte


class _EmbeddingOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_dict(self):
        return dict(self.kwargs)


class _FakeLanguageModel:
    pass


class _FakeVisionConfig:
    spatial_merge_size = 1


class _FakeConfig:
    image_token_id = 20
    vision_config = _FakeVisionConfig()


class _FakeModel:
    def __init__(self, *, inputs_embeds=None):
        self.calls = []
        self.config = _FakeConfig()
        self.language_model = _FakeLanguageModel()
        self.inputs_embeds = (
            inputs_embeds
            if inputs_embeds is not None
            else mx.zeros((1, 1, 2), dtype=mx.float32)
        )

    def get_input_embeddings(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _EmbeddingOutput(
            inputs_embeds=self.inputs_embeds,
            unused=None,
        )


def test_prepare_prompt_inputs_text_only_preserves_prompt_tokens():
    """Text-only requests keep caller tokenization as the source of truth."""
    prompt = prepare_prompt_inputs(
        prompt_tokens=[1, 2, 3],
        images_b64=None,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        config={},
    )

    assert prompt == PreparedPrompt(
        prompt_input_ids=[1, 2, 3],
        raw_inputs=None,
        image_spans=[],
    )


def test_prepare_prompt_inputs_builds_image_spans_from_processor_tokens(monkeypatch):
    """Image requests use mlx-vlm expanded IDs and hash each image run."""
    images = [_FakeImage(b"a"), _FakeImage(b"b")]

    def prepare_inputs(**kwargs):
        assert kwargs["prompts"] == "decoded prompt"
        assert kwargs["image_token_index"] == 20
        return {
            "input_ids": mx.array([[1, 20, 20, 2, 20, 3]], dtype=mx.int32),
            "pixel_values": mx.array([1], dtype=mx.float32),
        }

    monkeypatch.setattr(prompt_inputs_module, "convert_to_pil", lambda _b64: images)
    monkeypatch.setattr(prompt_inputs_module.mlx_vlm, "prepare_inputs", prepare_inputs)

    prompt = prepare_prompt_inputs(
        prompt_tokens=[9, 8, 7],
        images_b64=["image-a", "image-b"],
        tokenizer=_FakeTokenizer(),
        processor=object(),
        config={"image_token_id": 20},
    )

    assert prompt.prompt_input_ids == [1, 20, 20, 2, 20, 3]
    assert prompt.raw_inputs["pixel_values"].tolist() == [1.0]
    assert [(span.start, span.end) for span in prompt.image_spans] == [(1, 3), (4, 5)]
    assert prompt.image_spans[0].image_hash != prompt.image_spans[1].image_hash


def test_prepare_prompt_inputs_falls_back_to_whole_prompt_on_image_span_mismatch(
    monkeypatch,
):
    """Wrong image reuse is worse than losing a cache hit."""
    images = [_FakeImage(b"a"), _FakeImage(b"b")]
    monkeypatch.setattr(prompt_inputs_module, "convert_to_pil", lambda _b64: images)
    monkeypatch.setattr(
        prompt_inputs_module.mlx_vlm,
        "prepare_inputs",
        lambda **_kwargs: {
            "input_ids": mx.array([[1, 20, 20, 2]], dtype=mx.int32),
        },
    )

    prompt = prepare_prompt_inputs(
        prompt_tokens=[9],
        images_b64=["image-a", "image-b"],
        tokenizer=_FakeTokenizer(),
        processor=object(),
        config={"image_token_id": 20},
    )

    assert len(prompt.image_spans) == 1
    assert prompt.image_spans[0].start == 0
    assert prompt.image_spans[0].end == 4
    assert "|" in prompt.image_spans[0].image_hash


def test_build_prompt_kwargs_image_adds_qwen_position_state():
    """Image prefill exports Qwen MRoPE side state for chunked prefill."""
    input_ids = mx.array([[101, 20, 20, 102]], dtype=mx.int32)
    model = _FakeModel(inputs_embeds=mx.zeros((1, 4, 2), dtype=mx.float32))
    prepared_prompt = PreparedPrompt(
        prompt_input_ids=input_ids.squeeze(0).tolist(),
        raw_inputs={
            "input_ids": input_ids,
            "pixel_values": mx.array([1], dtype=mx.float32),
            "attention_mask": mx.array([[1, 1, 1, 1]], dtype=mx.int32),
            "image_grid_thw": mx.array([[1, 1, 2]], dtype=mx.int32),
        },
        image_spans=[PromptImageSpan(1, 3, "image")],
    )

    prompt_kwargs = build_prompt_kwargs(model, prepared_prompt)

    call_args, call_kwargs = model.calls[0]
    assert call_args[0] is input_ids
    assert call_args[1].tolist() == [1.0]
    assert call_kwargs["mask"].tolist() == [[1, 1, 1, 1]]
    assert prompt_kwargs["position_ids"].tolist() == [
        [[0, 1, 1, 3]],
        [[0, 1, 1, 3]],
        [[0, 1, 2, 3]],
    ]
    assert prompt_kwargs["rope_deltas"].item() == 0


def test_build_cached_prompt_kwargs_slices_image_embeds_and_position_ids(monkeypatch):
    """Image restores recompute full side state, then prefill only the suffix."""
    inputs_embeds = mx.arange(12, dtype=mx.float32).reshape(1, 6, 2)
    position_ids = mx.arange(18, dtype=mx.int32).reshape(3, 1, 6)

    def build_prompt_kwargs(_model, _prepared_prompt):
        return {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "rope_deltas": mx.array([5], dtype=mx.int32),
        }

    monkeypatch.setattr(
        prompt_inputs_module,
        "build_prompt_kwargs",
        build_prompt_kwargs,
    )

    prompt_kwargs = build_cached_prompt_kwargs(
        object(),
        PreparedPrompt(
            prompt_input_ids=list(range(6)),
            raw_inputs={"input_ids": mx.array([[0, 1, 2, 3, 4, 5]])},
            image_spans=[],
        ),
        cached_prefix_len=4,
        rope_deltas=None,
    )

    assert prompt_kwargs["inputs_embeds"].tolist() == inputs_embeds[:, 4:].tolist()
    assert prompt_kwargs["position_ids"].tolist() == position_ids[:, :, 4:].tolist()
    assert prompt_kwargs["rope_deltas"].tolist() == [5]
