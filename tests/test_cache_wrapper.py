import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper


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
        
    # TODO(christian-lms): write tests for cache shifting, which is high-level
    # implemented in cachewrapper and so belongs here


if __name__ == "__main__":
    unittest.main(verbosity=2)
