import typing
import unittest
from unittest import mock

from mlx_engine import generate


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
        return str(token)


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


class FakeStopResult:
    def __init__(self):
        self.status = "full_stop"
        self.stop_string = "stop"
        self.stop_tokens = [1]


class FakeStopProcessor:
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def process_token(self, token):  # noqa: ARG002
        return FakeStopResult()


class TestGenerateStopUnbounded(unittest.TestCase):
    def test_stop_string_detected_in_unbounded_plan(self):
        fake_plan = mock.Mock(mode="unbounded", chunk_size=None, reason="test")
        fake_stream = iter(
            [_GenerationResult(1, "hello stop"), _GenerationResult(0, "")]
        )

        with (
            mock.patch.object(
                generate, "plan_prefill_strategy", return_value=fake_plan, create=True
            ),
            mock.patch.object(
                generate, "stream_generate", return_value=fake_stream, autospec=True
            ),
            mock.patch.object(generate, "StopStringProcessor", FakeStopProcessor),
            mock.patch.object(generate, "get_eot_token_ids", return_value=set()),
        ):
            kit = _FakeModelKit()
            generator_fn = typing.cast(typing.Any, generate.create_generator)
            results = list(
                generator_fn(
                    kit,
                    prompt_tokens=[1, 2, 3],
                    stop_strings=["stop"],
                    prefill_mode="auto",
                )
            )

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].stop_condition)
        self.assertEqual(results[0].stop_condition.stop_reason, "stop_string")
        self.assertEqual(results[0].stop_condition.stop_string, "stop")
        self.assertEqual(results[0].stop_condition.stop_tokens, [1])


if __name__ == "__main__":
    unittest.main()
