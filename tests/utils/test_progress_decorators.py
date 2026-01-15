import unittest
from mlx_engine.utils.progress_decorators import (
    ratchet,
    throw_to_stop,
    mlx_lm_converter,
)
from mlx_engine.utils.prompt_progress_events import (
    PromptProgressBeginEvent,
    PromptProgressEvent,
)
from mlx_engine.cache_wrapper import StopPromptProcessing


def create_v2_callback(
    received_calls: list[PromptProgressBeginEvent | PromptProgressEvent], retval: bool
):
    """Create a V2 callback that records events and returns a fixed value."""

    def callback(
        event: PromptProgressBeginEvent | PromptProgressEvent, is_draft: bool
    ) -> bool:
        received_calls.append(event)
        return retval

    return callback


def create_begin_event(
    prefill_tokens_processed: int, total_prompt_tokens: int, cached_tokens: int
) -> PromptProgressBeginEvent:
    """Helper to create a BeginEvent."""
    return PromptProgressBeginEvent(
        prefill_tokens_processed=prefill_tokens_processed,
        total_prompt_tokens=total_prompt_tokens,
        cached_tokens=cached_tokens,
    )


def create_progress_event(prefill_tokens_processed: int) -> PromptProgressEvent:
    """Helper to create a PromptProgressEvent."""
    return PromptProgressEvent(prefill_tokens_processed=prefill_tokens_processed)


class TestRatchet(unittest.TestCase):
    def test_monotonic_progress(self):
        """Test that ratchet allows monotonic progress updates."""
        # Create events with increasing token counts
        inputs = [
            create_begin_event(0, 100, 0),
            create_progress_event(25),
            create_progress_event(50),
            create_progress_event(75),
        ]
        received_calls = []
        original_callback = create_v2_callback(received_calls, False)

        system_under_test = ratchet(original_callback)
        results = [system_under_test(event, False) for event in inputs]

        self.assertEqual(received_calls, inputs)
        self.assertEqual(results, [False] * 4)

    def test_non_monotonic_progress(self):
        """Test that ratchet disallows non-monotonic progress updates."""
        # Mix of forward and backward progress
        inputs = [
            create_begin_event(0, 100, 0),  # 0 - should pass
            create_progress_event(25),  # 25 - should pass
            create_progress_event(0),  # 0 - should be blocked (backwards)
            create_progress_event(50),  # 50 - should pass
            create_progress_event(30),  # 30 - should be blocked (backwards)
            create_progress_event(75),  # 75 - should pass
            create_progress_event(60),  # 60 - should be blocked (backwards)
            create_progress_event(100),  # 100 - should pass
            create_progress_event(99),  # 99 - should be blocked (backwards)
        ]
        # We construct the return value so wrapped callback returns False when original
        # is called, True otherwise (when blocked)
        expected_results = [False, False, True, False, True, False, True, False, True]
        expected_calls = [
            event for (event, result) in zip(inputs, expected_results) if not result
        ]
        received_calls = []
        original_callback = create_v2_callback(received_calls, False)

        system_under_test = ratchet(original_callback)
        results = [system_under_test(event, False) for event in inputs]

        self.assertEqual(received_calls, expected_calls)
        self.assertEqual(results, expected_results)


class TestThrowToStop(unittest.TestCase):
    def test_callback_returns_true(self):
        """Test that throw_to_stop continues when callback returns True."""
        inputs = [
            create_begin_event(0, 100, 0),
            create_progress_event(25),
            create_progress_event(50),
        ]
        received_calls = []
        original_callback = create_v2_callback(received_calls, True)

        system_under_test = throw_to_stop(original_callback)
        results = [system_under_test(event, False) for event in inputs]

        self.assertEqual(received_calls, inputs)
        self.assertEqual(results, [True] * 3)

    def test_callback_returns_false_raises_exception(self):
        """Test that throw_to_stop raises StopPromptProcessing when callback returns False."""
        input_event = create_progress_event(25)
        received_calls = []
        original_callback = create_v2_callback(received_calls, False)

        system_under_test = throw_to_stop(original_callback)
        with self.assertRaises(StopPromptProcessing):
            system_under_test(input_event, False)

        self.assertEqual(len(received_calls), 1)
        self.assertEqual(received_calls[0], input_event)


class TestMlxLmConverter(unittest.TestCase):
    """Test the mlx_lm_converter decorator that converts token counts to V2 events."""

    def test_converter_with_emit_begin_true(self):
        """Test that mlx_lm_converter emits BeginEvent when emit_begin_event=True."""
        received_calls = []
        original_callback = create_v2_callback(received_calls, True)

        # Create converter with emit_begin_event=True
        system_under_test = mlx_lm_converter(original_callback, emit_begin_event=True)

        # Call the converter (takes tokens_processed, total_tokens)
        result = system_under_test(0, 100)
        self.assertIsNone(result)

        # Should have received only a BeginEvent
        self.assertEqual(len(received_calls), 1)

        begin_event = received_calls[0]
        self.assertIsInstance(begin_event, PromptProgressBeginEvent)
        self.assertEqual(begin_event.prefill_tokens_processed, 0)
        self.assertEqual(begin_event.total_prompt_tokens, 100)
        self.assertEqual(begin_event.cached_tokens, 0)

    def test_converter_with_emit_begin_false(self):
        """Test that mlx_lm_converter emits only ProgressEvent when emit_begin_event=False."""
        received_calls = []
        original_callback = create_v2_callback(received_calls, True)

        # Create converter with emit_begin_event=False (default)
        system_under_test = mlx_lm_converter(original_callback, emit_begin_event=False)

        # Call the converter
        result = system_under_test(50, 100)
        self.assertIsNone(result)

        # Should have received only a ProgressEvent
        self.assertEqual(len(received_calls), 1)
        event = received_calls[0]
        self.assertIsInstance(event, PromptProgressEvent)
        self.assertEqual(event.prefill_tokens_processed, 50)

    def test_converter_sequence_with_begin(self):
        """Test a sequence of calls with BeginEvent on first call."""
        received_calls = []
        original_callback = create_v2_callback(received_calls, True)

        # Create converter with emit_begin_event=True
        system_under_test = mlx_lm_converter(original_callback, emit_begin_event=True)

        # Simulate a typical progress sequence (only passing tokens_processed, total_tokens)
        inputs = [
            (0, 100),  # First call - will emit Begin + Progress
            (25, 100),  # Progress
            (50, 100),  # Progress
            (100, 100),  # Progress (complete)
        ]

        for tokens_processed, total_tokens in inputs:
            system_under_test(tokens_processed, total_tokens)

        # Verify we got the right sequence
        # First call emits Begin, rest emit Progress
        self.assertEqual(len(received_calls), 4)

        # First should be BeginEvent (only on first call)
        self.assertIsInstance(received_calls[0], PromptProgressBeginEvent)
        self.assertEqual(received_calls[0].prefill_tokens_processed, 0)
        self.assertEqual(received_calls[0].total_prompt_tokens, 100)

        # Rest should be ProgressEvents
        expected_tokens = [25, 50, 100]
        for index, expected_token_count in enumerate(expected_tokens, start=1):
            self.assertIsInstance(received_calls[index], PromptProgressEvent)
            self.assertEqual(
                received_calls[index].prefill_tokens_processed, expected_token_count
            )

    def test_converter_sequence_without_begin(self):
        """Test a sequence of calls without BeginEvent."""
        received_calls = []
        original_callback = create_v2_callback(received_calls, True)

        # Create converter with emit_begin_event=False
        system_under_test = mlx_lm_converter(original_callback, emit_begin_event=False)

        # Simulate progress sequence
        inputs = [(0, 100), (25, 100), (50, 100), (100, 100)]

        for tokens_processed, total_tokens in inputs:
            system_under_test(tokens_processed, total_tokens)

        # Should only get ProgressEvents, no BeginEvent
        self.assertEqual(len(received_calls), 4)

        expected_tokens = [0, 25, 50, 100]
        for index, expected_token_count in enumerate(expected_tokens):
            self.assertIsInstance(received_calls[index], PromptProgressEvent)
            self.assertEqual(
                received_calls[index].prefill_tokens_processed, expected_token_count
            )
