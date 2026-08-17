from types import SimpleNamespace

import mlx_engine.generate as generate_module


def test_batched_vision_generation_installs_muse_glimmer_processor(monkeypatch):
    context = object()
    processor = object()

    class FakeVisionModelKit:
        def __init__(self):
            self.tokenizer = SimpleNamespace()
            self.model_type = "muse_glimmer"
            self.generate_args = None

        def generate(self, **kwargs):
            self.generate_args = kwargs
            return iter(())

    monkeypatch.setattr(generate_module, "BatchedVisionModelKit", FakeVisionModelKit)
    monkeypatch.setattr(
        generate_module,
        "create_muse_glimmer_tool_context_from_prompt",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        generate_module,
        "create_muse_glimmer_tool_logits_processor",
        lambda **_kwargs: processor,
    )

    model_kit = FakeVisionModelKit()
    assert (
        list(
            generate_module._batched_generation(
                model_kit,
                [1, 2, 3],
                request_id="request",
            )
        )
        == []
    )
    assert model_kit.generate_args["logits_processors"] == [processor]
