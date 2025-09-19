import unittest
from mlx_engine.utils.progress_decorators import ratchet, throw_to_stop, token_count
from mlx_engine.cache_wrapper import StopPromptProcessing
from typing import Callable, TypeVar


def create_callback(received_calls: list[float], retval: bool):
    def callback(val: float) -> bool:
        received_calls.append(val)
        return retval

    return callback


def execute_calls(
    system_under_test: Callable[[float], bool], inputs: list[float]
) -> list[bool]:
    return [system_under_test(i) for i in inputs]


T = TypeVar("T")


def unwrap_optional(value: T | None) -> T:
    if value is None:
        raise RuntimeError("Value cannot be None")
    return value


class TestRatchet(unittest.TestCase):
    def test_none_callback_returns_none(self):
        """Test that ratchet returns None when given None."""
        result = ratchet(None)
        self.assertIsNone(result)

    def test_monotonic_progress(self):
        """Test that ratchet allows monotonic progress updates."""
        inputs = [25.0, 50.0, 75.0]
        received_calls = []
        original_callback = create_callback(received_calls, False)

        system_under_test = unwrap_optional(ratchet(original_callback))
        results = [system_under_test(i) for i in inputs]

        self.assertEqual(received_calls, inputs)
        self.assertEqual(results, [False] * 3)

    def test_non_monotonic_progress(self):
        """Test that ratchet disallows non-monotonic progress updates."""
        inputs = [0.0, 25.0, 0.0, 50.0, 30.0, 75.0, 60.0, 100.0, 99.9]
        # We construct the return value of the original_callback so that the wrapped callback will
        # return False when original_callback is called, and True otherwise.
        expected_results = [False, False, True, False, True, False, True, False, True]
        expected_calls = [
            input for (input, result) in zip(inputs, expected_results) if not result
        ]
        received_calls = []
        original_callback = create_callback(received_calls, False)

        system_under_test = unwrap_optional(ratchet(original_callback))
        results = [system_under_test(i) for i in inputs]

        self.assertEqual(received_calls, expected_calls)
        self.assertEqual(results, expected_results)


class TestThrowToStop(unittest.TestCase):
    def test_none_callback_returns_none(self):
        """Test that throw_to_stop returns None when given None."""
        result = throw_to_stop(None)
        self.assertIsNone(result)

    def test_callback_returns_true(self):
        """Test that throw_to_stop continues when callback returns True."""
        inputs = [25.0, 50.0, 75.0]
        received_calls = []
        original_callback = create_callback(received_calls, True)

        system_under_test = unwrap_optional(throw_to_stop(original_callback))
        results = [system_under_test(i) for i in inputs]

        self.assertEqual(received_calls, inputs)
        self.assertEqual(results, [True] * 3)

    def test_callback_returns_false_raises_exception(self):
        """Test that throw_to_stop raises StopPromptProcessing when callback returns False."""
        input = 25.0
        received_calls = []
        original_callback = create_callback(received_calls, False)

        system_under_test = unwrap_optional(throw_to_stop(original_callback))
        with self.assertRaises(StopPromptProcessing):
            system_under_test(input)

        self.assertEqual(received_calls, [input])


class TestTokenCount(unittest.TestCase):
    def test_none_callback_returns_none(self):
        """Test that token_count returns None when given None."""
        result = token_count(None)
        self.assertIsNone(result)

    def test_token_count_callback(self):
        """Test that token_count calls the callback with correct token counts."""
        inputs = [(0, 30), (10, 30), (20, 30), (30, 30)]
        expected_calls = [input[0] / input[1] * 100.0 for input in inputs]
        received_calls = []
        original_callback = create_callback(received_calls, True)

        system_under_test = unwrap_optional(token_count(original_callback))
        results = [system_under_test(*i) for i in inputs]

        for received, expected in zip(received_calls, expected_calls):
            self.assertAlmostEqual(received, expected)
        self.assertEqual(results, [None] * len(inputs))
