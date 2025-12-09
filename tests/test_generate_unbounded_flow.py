import typing
import unittest
from unittest import mock

from mlx_engine import generate
from mlx_engine.model_kit.model_kit import ModelKit


class _Detokenizer:
    def __init__(self):
        self.last_segment = ""

    def finalize(self):
        self.last_segment = ""


class _Tokenizer:
    def __init__(self):
        self.eos_token_ids = {0}
        self.detokenizer = _Detokenizer()
        self._tokenizer = None

    def decode(self, token):
        return f"{token}"


class _GenerationResult:
    def __init__(self, token, text="", from_draft=False):
        self.token = token
        self.text = text
        self.from_draft = from_draft
        self.logprobs = {token: 0.0}


class _FakeModelKit:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.model = object()
        self.model_type = "text"
        self.max_kv_size = None
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.draft_model = None

    def process_prompt(
        self,
        prompt_tokens,
        images_b64,
        prompt_progress_callback,
        generate_args,
        max_image_size,
        speculative_decoding_toggle,
    ):
        return prompt_tokens, None

    def is_cross_prompt_cache_active(self):
        return False

    def record_token_to_cache(self, token):
        return None

    def tokenize(self, prompt):
        return [1, 2, 3]


class TestGenerateUnboundedFlow(unittest.TestCase):
    def test_create_generator_uses_prefill_plan_when_provided(self):
        fake_plan = mock.Mock(mode="chunked", chunk_size=4096, reason="test")
        fake_stream = iter([_GenerationResult(1, "a"), _GenerationResult(0, "")])

        with (
            mock.patch.object(
                generate, "plan_prefill_strategy", return_value=fake_plan, create=True
            ) as plan,
            mock.patch.object(
                generate, "stream_generate", return_value=fake_stream, autospec=True
            ) as stream,
            mock.patch.object(generate, "get_eot_token_ids", return_value=set()),
        ):
            kit = _FakeModelKit()
            generator_fn = typing.cast(typing.Any, generate.create_generator)
            list(
                generator_fn(
                    kit,
                    prompt_tokens=[1, 2, 3],
                    prompt_progress_callback=None,
                    **{
                        "prefill_mode": "auto",
                        "performance_profile": "m3_ultra_512",
                        "available_mem_gb": 512,
                    },
                )
            )

        plan.assert_called()
        self.assertEqual(stream.call_args.kwargs.get("prefill_step_size"), 4096)


if __name__ == "__main__":
    unittest.main()
