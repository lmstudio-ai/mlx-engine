import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper
from mlx_engine.cache import ShiftingKVCache
from tests.test_cache_generic import TestCache
from tests.utils import DummyModel


class TestCacheWrapper(TestCache):
    def test_find_matching_sequence_length_with_mismatch(self):
        """Test when there's a mismatch in the tokens"""
        # Create two arrays with a known common prefix [1, 2, 3]
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 6, 7])  # Mismatch at index 3
        num_tokens_to_exclude = 1

        print("\nTest with mismatch:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens, num_tokens_to_exclude=num_tokens_to_exclude
        )
        self.assertEqual(result, 3)  # Should find 3 matching tokens

    def test_find_matching_sequence_length_all_match(self):
        """Test when all tokens match"""
        # Create two identical arrays
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])  # All tokens match
        num_tokens_to_exclude = 1

        print("\nTest with all matching:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens, num_tokens_to_exclude=num_tokens_to_exclude
        )
        self.assertEqual(
            result, 4
        )  # Should find 4 matching tokens (5-1 due to num_tokens_to_exclude)

    def test_find_matching_sequence_length_no_match(self):
        """Test when no tokens match"""
        # Create two arrays with no common prefix
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([6, 7, 8, 9, 10])
        num_tokens_to_exclude = 1

        print("\nTest with no matching tokens:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens, num_tokens_to_exclude=num_tokens_to_exclude
        )
        self.assertEqual(result, 0)  # No matching tokens should return 0

    def test_find_matching_sequence_length_offset_starts(self):
        """Test when the current tokens start with a different offset"""
        # Create two arrays where the current tokens start with a different offset
        current_tokens = mx.array([2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])
        num_tokens_to_exclude = 1

        print("\nTest with offset starts:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens,
            prompt_tokens,
            start2=1,
            num_tokens_to_exclude=num_tokens_to_exclude,
        )
        self.assertEqual(result, 3)

    def test_find_matching_sequence_length_more_offsets(self):
        """Test when the current tokens have more offsets"""
        # Create two arrays where the current tokens have more offsets
        current_tokens = mx.array([1, 2, 3, 4, 5, 6])
        prompt_tokens = mx.array([0, 9, 10, 3, 4, 7, 8])

        print("\nTest with more offsets:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens
        )
        self.assertEqual(result, 0)

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens,
            prompt_tokens,
            start1=2,
            start2=3,
        )
        self.assertEqual(result, 2)

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens,
            prompt_tokens,
            start1=2,
            start2=3,
            num_tokens_to_exclude=1,
        )
        self.assertEqual(result, 2)  # there are leftovers anyway

    def test_record_generated_token_loops(self):
        cache = CacheWrapper(
            model=DummyModel(),
            max_kv_size=5,
            keep=2,
        )
        cache.tokens = mx.array([])
        cache.record_generated_token(1)
        cache.record_generated_token(2)
        cache.record_generated_token(3)
        cache.record_generated_token(4)
        cache.record_generated_token(5)
        self.assertListEqual(
            cache.tokens.tolist(),
            [1, 2, 3, 4, 5],
        )
        cache.record_generated_token(6)
        self.assertListEqual(
            cache.tokens.tolist(),
            [1, 2, 4, 5, 6],
        )

    def test_cache_reuse(self):
        cache = CacheWrapper(DummyModel(), 10)
        cache.cache = ShiftingKVCache(self._rope, max_size=10, keep=2)

        # set up pretend cache
        cached_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cache_kv = self.make_random_kv(10)
        cache.tokens = cached_tokens
        cache.cache.update_and_fetch(cache_kv, cache_kv)

        prompt_tokens = mx.array([1, 2, 4, 7, 8, 9, 11])
        prefix_len = cache._find_matching_sequence_length(
            cached_tokens, prompt_tokens, 0
        )
        self.assertEqual(prefix_len, 2)

        total_reused = cache._truncate_cache(
            prompt_tokens=prompt_tokens,
            common_prefix_len=prefix_len,
            non_prefix_reuse_min_seq_len=1,
        )

        should_be_tokens = mx.array([1, 2, 4, 7, 8, 9])

        def idx(v, a, b):
            return v[:, :, a:b, :]

        should_be_kv = mx.concatenate(
            [idx(cache_kv, 0, 2), idx(cache_kv, 3, 4), idx(cache_kv, 6, 9)]
        )

        self.assertEqual(total_reused, 4)
        self.assertArrEqual(cache.tokens, should_be_tokens)
        self.assertArrEqual(cache.cache.keys, should_be_kv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
