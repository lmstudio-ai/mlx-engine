import unittest
from pathlib import Path

from mlx_engine.utils.disable_hf_download import _original_snapshot_download
import mlx_lm

from mlx_engine.stop_string_processor import StopStringProcessor


class TestStopStringProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any necessary resources that can be shared across all tests."""
        # use Llama-3.1 tokenizer for testing
        cls.tokenizer = cls.download_tokenizer(
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        )

    # followed pattern from mlx-examples
    # https://github.com/ml-explore/mlx-examples/blob/cfc29c29f45372c78876335a44b0c99ab6565ae0/llms/tests/test_tokenizers.py#L17
    @staticmethod
    def download_tokenizer(repo):
        path = Path(
            _original_snapshot_download(
                repo_id=repo,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "tokenizer.model",
                ],
            )
        )
        return mlx_lm.tokenizer_utils.load(path)

    def process_tokens(self, stop_strings, input_string):
        """Helper method to process tokens and collect results"""
        processor = StopStringProcessor(stop_strings, self.tokenizer)
        input_tokens = self.tokenizer.encode(input_string)
        results = []
        for token in input_tokens:
            result = processor.process_token(token)
            results.append(result)
            if result.status == "full_stop":
                break
        return results

    def test_stop_string_processor_simple(self):
        results = self.process_tokens(["of"], "The objective of chess")

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "of")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" of"))

    def test_stop_string_at_start(self):
        results = self.process_tokens(["Hello"], "Hello world")
        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "Hello")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode("Hello"))

    def test_stop_string_at_end(self):
        results = self.process_tokens(["world"], "Hello world")
        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "world")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" world"))

    def test_case_sensitivity(self):
        results = self.process_tokens(["Stop"], "This is a STOP sign")
        self.assertEqual(results[-1].status, "no_match")

    def test_stop_string_with_special_characters(self):
        results = self.process_tokens(["\n"], "Hello\nworld")
        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "\n")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode("\n"))

    def test_unicode_stop_strings(self):
        results = self.process_tokens(["é", "ñ", "北京"], "Hello 北京 é ñ")
        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "北京")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" 北京"))

    def test_stop_string_processor_no_match(self):
        results = self.process_tokens(["other"], "The objective of chess")

        for i, result in enumerate(results):
            self.assertEqual(
                result.status,
                "no_match",
                f"Result at position {i} has status '{result.status}' instead of 'no_match'",
            )

    def test_stop_string_processor_long_no_match(self):
        results = self.process_tokens(
            ["The objective of checkers"], "The objective of chess"
        )

        for i, result in enumerate(results[:-1]):
            self.assertEqual(
                result.status,
                "partial_match",
                f"Result at position {i} has status '{result.status}' instead of 'partial_match'",
            )
        self.assertEqual(results[-1].status, "no_match")

    def test_stop_string_processor_mid_word(self):
        results = self.process_tokens(["cti"], "The objective of chess")

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "cti")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" objective"))

    def test_stop_string_processor_multi_token_multi_word(self):
        results = self.process_tokens(["objective of"], "The objective of chess")

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "objective of")
        self.assertEqual(
            results[-1].stop_tokens, self.tokenizer.encode(" objective of")
        )

    def test_stop_string_processor_multi_token_multi_token_single_char(self):
        results = self.process_tokens(["🌟"], "The objective 🌟 of chess")

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "🌟")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" 🌟"))

        self.assertEqual(results[-3].status, "multi_byte")
        self.assertEqual(results[-2].status, "multi_byte")

    def test_multiple_stop_strings(self):
        results = self.process_tokens(
            ["of", "chess", "objective"], "The objective of chess"
        )

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "objective")
        self.assertEqual(results[-1].stop_tokens, self.tokenizer.encode(" objective"))

    def test_overlapping_stop_strings(self):
        results = self.process_tokens(
            ["objective of", "of chess"], "The objective of chess"
        )

        self.assertEqual(results[-1].status, "full_stop")
        self.assertEqual(results[-1].stop_string, "objective of")
        self.assertEqual(
            results[-1].stop_tokens, self.tokenizer.encode(" objective of")
        )

    def test_empty_stop_strings_list_raises(self):
        with self.assertRaises(ValueError):
            StopStringProcessor([], self.tokenizer)

    def test_non_string_stop_string_raises(self):
        stop_strings = ["valid", 123]
        with self.assertRaises(TypeError):
            StopStringProcessor(stop_strings, self.tokenizer)

    def test_none_stop_string_raises(self):
        stop_strings = ["valid", None]
        with self.assertRaises(TypeError):
            StopStringProcessor(stop_strings, self.tokenizer)

    def test_empty_stop_string_raises(self):
        stop_strings = ["valid", ""]
        with self.assertRaises(ValueError):
            StopStringProcessor(stop_strings, self.tokenizer)


# Pytest wrapper to run the unittest with proper environment
def test_stop_string_processor_with_unittest():
    """Run stop string processor tests using unittest to avoid mocking conflicts."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_stop_string_processor.TestStopStringProcessor",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/miter/repo/third/mlx-engine",
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert False, f"Tests failed with return code {result.returncode}"
