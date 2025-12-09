import unittest
from unittest import mock
import sys

# Mock outlines module to avoid dependency issues
mock_outlines = mock.MagicMock()
mock_outlines.processors = mock.MagicMock()
mock_outlines.processors.structured = mock.MagicMock()
mock_outlines.processors.structured.JSONLogitsProcessor = mock.MagicMock()
mock_outlines.models = mock.MagicMock()
mock_outlines.models.transformers = mock.MagicMock()
sys.modules["outlines"] = mock_outlines
sys.modules["outlines.processors"] = mock_outlines.processors
sys.modules["outlines.processors.structured"] = mock_outlines.processors.structured
sys.modules["outlines.models"] = mock_outlines.models
sys.modules["outlines.models.transformers"] = mock_outlines.models.transformers

import mlx_engine as generate


class TestCliFlagHelp(unittest.TestCase):
    def test_invalid_profile_raises_value_error(self):
        with self.assertRaises(ValueError):
            generate.select_profile_for_hardware(  # type: ignore[attr-defined]
                None, requested="bad-profile", available_mem_gb=16
            )

    def test_help_includes_prefill_flags(self):
        parser = generate.cli_parser()  # type: ignore[attr-defined]
        help_text = parser.format_help()
        for flag in [
            "--prefill-mode",
            "--profile",
            "--cache-slots",
            "--kv-branching",
            "--progress-interval-ms",
            "--max-prefill-tokens",
        ]:
            self.assertIn(flag, help_text)


if __name__ == "__main__":
    unittest.main()
