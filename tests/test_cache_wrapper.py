import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper, StopPromptProcessing
from tests.shared import model_getter
from mlx_engine.generate import load_model, tokenize


class TestCacheWrapper(unittest.TestCase):
    def test_find_common_prefix_with_mismatch(self):
        """Test when there's a mismatch in the tokens"""
        # Create two arrays with a known common prefix [1, 2, 3]
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 6, 7])  # Mismatch at index 3
        num_tokens_to_exclude = 1

        print("\nTest with mismatch:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(result, 3)  # Should find 3 matching tokens

    def test_find_common_prefix_all_match(self):
        """Test when all tokens match"""
        # Create two identical arrays
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])  # All tokens match
        num_tokens_to_exclude = 1

        print("\nTest with all matching:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(
            result, 4
        )  # Should find 4 matching tokens (5-1 due to num_tokens_to_exclude)

    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
        num_tokens_to_exclude = 1
        model_kit.cache_wrapper = CacheWrapper(
            model_kit.model,
            max_kv_size=4096,
            chunk_size=chunk_size,
        )

        long_prompt = (
            "This is a test prompt that needs to be long enough to require multiple chunks for processing. "
            * 50
        )
        prompt_tokens = mx.array(tokenize(model_kit, long_prompt))
        tokens_to_process = len(prompt_tokens) - num_tokens_to_exclude
        # ceiling division
        expected_chunks = (tokens_to_process + chunk_size - 1) // chunk_size

        # First attempt: Progress callback that cancels after a few updates
        first_progress_calls = []

        def cancelling_progress_callback(progress):
            first_progress_calls.append(progress)
            if len(first_progress_calls) >= 3:
                return False
            return True

        with self.assertRaises(StopPromptProcessing):
            model_kit.cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                prompt_progress_callback=cancelling_progress_callback,
                num_tokens_to_exclude=1,
            )
        first_attempt_progress_calls = len(first_progress_calls)

        # Second attempt: Progress callback that doesn't cancel
        second_progress_calls = []

        def non_cancelling_progress_callback(progress):
            second_progress_calls.append(progress)
            return True

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            prompt_progress_callback=non_cancelling_progress_callback,
            num_tokens_to_exclude=1,
        )
        second_attempt_progress_calls = len(second_progress_calls)

        self.assertEqual(
            second_attempt_progress_calls,
            # +1 for the final 100% callback, +1 for the duplicate 0% callback
            expected_chunks - first_attempt_progress_calls + 2,
        )

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
