import unittest
from typing import Optional
from mlx_engine.utils.prompt_progress_reporter import (
    ThrowToStopReporter,
    MlxLmReporterAdapter,
)
from mlx_engine.cache_wrapper import StopPromptProcessing
from tests.shared import RecordingReporter


class MockReporter(RecordingReporter):
    """A mock reporter that records and prints calls, with a configurable return value."""

    def __init__(self, return_value: bool = True):
        super().__init__()
        self.return_value = return_value

    def begin(
        self,
        is_draft: bool,
        cached_tokens: int,
        total_prompt_tokens: int,
        prefill_tokens_processed: int,
    ) -> bool:
        super().begin(
            is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed
        )
        return self.return_value

    def update(self, is_draft: bool, prefill_tokens_processed: int) -> bool:
        super().update(is_draft, prefill_tokens_processed)
        return self.return_value

    def finish(
        self, is_draft: bool, prefill_tokens_processed: Optional[int] = None
    ) -> bool:
        super().finish(is_draft, prefill_tokens_processed)
        return self.return_value


class TestThrowToStopReporter(unittest.TestCase):
    def test_continues_when_inner_returns_true(self):
        """Test that ThrowToStopReporter continues when inner returns True."""
        inner = MockReporter(return_value=True)
        reporter = ThrowToStopReporter(inner)

        # These should not raise
        reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        reporter.update(is_draft=False, prefill_tokens_processed=25)
        reporter.update(is_draft=False, prefill_tokens_processed=50)
        reporter.finish(is_draft=False)

        self.assertEqual(len(inner.events), 4)

    def test_raises_on_begin_false(self):
        """Test that ThrowToStopReporter raises when begin returns False."""
        inner = MockReporter(return_value=False)
        reporter = ThrowToStopReporter(inner)

        with self.assertRaises(StopPromptProcessing):
            reporter.begin(
                is_draft=False,
                cached_tokens=0,
                total_prompt_tokens=100,
                prefill_tokens_processed=0,
            )

    def test_raises_on_update_false(self):
        """Test that ThrowToStopReporter raises when update returns False."""
        inner = MockReporter(return_value=False)
        reporter = ThrowToStopReporter(inner)

        with self.assertRaises(StopPromptProcessing):
            reporter.update(is_draft=False, prefill_tokens_processed=25)

        self.assertEqual(len(inner.events), 1)
        self.assertEqual(inner.events[0]["prefill_tokens_processed"], 25)

    def test_raises_on_finish_false(self):
        """Test that ThrowToStopReporter raises when finish returns False."""
        inner = MockReporter(return_value=False)
        reporter = ThrowToStopReporter(inner)

        with self.assertRaises(StopPromptProcessing):
            reporter.finish(is_draft=False)


class TestMlxLmReporterAdapter(unittest.TestCase):
    """Test the MlxLmReporterAdapter that converts token counts to reporter method calls."""

    def test_adapter_with_emit_begin_true(self):
        """Test that adapter emits begin when emit_begin=True."""
        inner = MockReporter(return_value=True)
        adapter = MlxLmReporterAdapter(inner, emit_begin=True)

        # First call should emit begin
        adapter(0, 100)

        self.assertEqual(len(inner.events), 1)
        self.assertEqual(inner.events[0]["type"], "begin")
        self.assertEqual(inner.events[0]["cached_tokens"], 0)
        self.assertEqual(inner.events[0]["total_prompt_tokens"], 100)

    def test_adapter_with_emit_begin_false(self):
        """Test that adapter emits only update when emit_begin=False."""
        inner = MockReporter(return_value=True)
        adapter = MlxLmReporterAdapter(inner, emit_begin=False)

        # Call should emit update, not begin
        adapter(50, 100)

        self.assertEqual(len(inner.events), 1)
        self.assertEqual(inner.events[0]["type"], "update")
        self.assertEqual(inner.events[0]["prefill_tokens_processed"], 50)

    def test_adapter_sequence_with_begin(self):
        """Test a sequence of calls with begin on first call."""
        inner = MockReporter(return_value=True)
        adapter = MlxLmReporterAdapter(inner, emit_begin=True)

        # Simulate a typical progress sequence
        adapter(0, 100)  # First call - emits begin
        adapter(25, 100)  # Update
        adapter(50, 100)  # Update
        adapter(100, 100)  # Complete - emits finish

        # Verify sequence: begin, update, update, finish
        self.assertEqual(len(inner.events), 4)
        self.assertEqual(inner.events[0]["type"], "begin")
        self.assertEqual(inner.events[1]["type"], "update")
        self.assertEqual(inner.events[1]["prefill_tokens_processed"], 25)
        self.assertEqual(inner.events[2]["type"], "update")
        self.assertEqual(inner.events[2]["prefill_tokens_processed"], 50)
        self.assertEqual(inner.events[3]["type"], "finish")
        self.assertEqual(inner.events[3]["prefill_tokens_processed"], 100)

    def test_adapter_sequence_without_begin(self):
        """Test a sequence of calls without begin."""
        inner = MockReporter(return_value=True)
        adapter = MlxLmReporterAdapter(inner, emit_begin=False)

        # Simulate progress sequence
        adapter(0, 100)
        adapter(25, 100)
        adapter(50, 100)
        adapter(100, 100)  # Complete - emits finish

        # Should get: update, update, update, finish
        self.assertEqual(len(inner.events), 4)
        self.assertEqual(inner.events[0]["type"], "update")
        self.assertEqual(inner.events[0]["prefill_tokens_processed"], 0)
        self.assertEqual(inner.events[1]["type"], "update")
        self.assertEqual(inner.events[1]["prefill_tokens_processed"], 25)
        self.assertEqual(inner.events[2]["type"], "update")
        self.assertEqual(inner.events[2]["prefill_tokens_processed"], 50)
        self.assertEqual(inner.events[3]["type"], "finish")
        self.assertEqual(inner.events[3]["prefill_tokens_processed"], 100)

    def test_adapter_ignores_calls_after_finish(self):
        """Test that adapter ignores calls after finish has been called."""
        inner = MockReporter(return_value=True)
        adapter = MlxLmReporterAdapter(inner, emit_begin=False)

        adapter(50, 100)
        adapter(100, 100)  # This triggers finish
        adapter(100, 100)  # Should be ignored
        adapter(150, 100)  # Should be ignored

        # Only 2 calls: update at 50, finish at 100
        self.assertEqual(len(inner.events), 2)
