import unittest
import mlx.core as mx
from mlx_engine.recurrent_checkpoint_store import RecurrentCheckpointStore


class MockCacheLayer:
    """Mock cache layer that mimics the interface of MLX cache layers."""

    def __init__(self, offset: int = 0, data: float = 0.0):
        self.offset = offset
        self._data = mx.array([data])

    @property
    def state(self):
        return [self._data]

    def __eq__(self, other):
        if not isinstance(other, MockCacheLayer):
            return False
        return self.offset == other.offset and mx.array_equal(self._data, other._data)


def make_mock_cache(num_layers: int = 2, offset: int = 0, data: float = 0.0):
    """Create a list of mock cache layers."""
    return [MockCacheLayer(offset=offset, data=data + i) for i in range(num_layers)]


class TestRecurrentCheckpointStore(unittest.TestCase):
    def test_save_and_find_exact_match(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)
        tokens = mx.array([1, 2, 3, 4, 5])
        cache = make_mock_cache(offset=5, data=1.0)

        store.save(tokens, cache)
        result = store.find_longest_prefix(tokens)

        self.assertIsNotNone(result)
        prefix_len, restored_cache = result
        self.assertEqual(prefix_len, 5)
        self.assertEqual(len(restored_cache), 2)

    def test_find_longest_prefix(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)

        # Save checkpoints at different positions
        tokens_3 = mx.array([1, 2, 3])
        cache_3 = make_mock_cache(offset=3, data=3.0)
        store.save(tokens_3, cache_3)

        tokens_5 = mx.array([1, 2, 3, 4, 5])
        cache_5 = make_mock_cache(offset=5, data=5.0)
        store.save(tokens_5, cache_5)

        tokens_2 = mx.array([1, 2])
        cache_2 = make_mock_cache(offset=2, data=2.0)
        store.save(tokens_2, cache_2)

        # Query with tokens that extend beyond the longest checkpoint
        query = mx.array([1, 2, 3, 4, 5, 6, 7])
        result = store.find_longest_prefix(query)

        self.assertIsNotNone(result)
        prefix_len, restored_cache = result
        self.assertEqual(prefix_len, 5)  # Should match the 5-token checkpoint

    def test_no_match_returns_none(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)

        tokens = mx.array([1, 2, 3])
        cache = make_mock_cache(offset=3)
        store.save(tokens, cache)

        # Completely different tokens
        query = mx.array([10, 20, 30])
        result = store.find_longest_prefix(query)
        self.assertIsNone(result)

    def test_empty_store_returns_none(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)
        query = mx.array([1, 2, 3])
        result = store.find_longest_prefix(query)
        self.assertIsNone(result)

    def test_lru_eviction(self):
        store = RecurrentCheckpointStore(max_checkpoints=3)

        # Save 3 checkpoints (at capacity)
        for i in range(3):
            tokens = mx.array([i, i + 1, i + 2])
            cache = make_mock_cache(offset=3, data=float(i))
            store.save(tokens, cache)

        self.assertEqual(len(store), 3)

        # Save a 4th — should evict the oldest (i=0)
        tokens_new = mx.array([100, 101, 102])
        cache_new = make_mock_cache(offset=3, data=100.0)
        store.save(tokens_new, cache_new)

        self.assertEqual(len(store), 3)

        # The first checkpoint (i=0) should be evicted
        query_evicted = mx.array([0, 1, 2])
        result = store.find_longest_prefix(query_evicted)
        self.assertIsNone(result)

        # The new checkpoint should be present
        result_new = store.find_longest_prefix(tokens_new)
        self.assertIsNotNone(result_new)

    def test_lru_access_order(self):
        store = RecurrentCheckpointStore(max_checkpoints=3)

        tokens_a = mx.array([1, 2, 3])
        tokens_b = mx.array([4, 5, 6])
        tokens_c = mx.array([7, 8, 9])

        store.save(tokens_a, make_mock_cache(data=1.0))
        store.save(tokens_b, make_mock_cache(data=2.0))
        store.save(tokens_c, make_mock_cache(data=3.0))

        # Access tokens_a to make it most-recently-used
        store.find_longest_prefix(tokens_a)

        # Add a new entry — should evict tokens_b (LRU), not tokens_a
        tokens_d = mx.array([10, 11, 12])
        store.save(tokens_d, make_mock_cache(data=4.0))

        # tokens_b should be evicted
        result_b = store.find_longest_prefix(tokens_b)
        self.assertIsNone(result_b)

        # tokens_a should still be present
        result_a = store.find_longest_prefix(tokens_a)
        self.assertIsNotNone(result_a)

    def test_deepcopy_independence(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)
        tokens = mx.array([1, 2, 3])
        original_cache = make_mock_cache(offset=3, data=1.0)

        store.save(tokens, original_cache)

        # Get a restored copy
        result = store.find_longest_prefix(tokens)
        self.assertIsNotNone(result)
        _, restored_cache = result

        # Modify the restored cache
        restored_cache[0]._data = mx.array([999.0])

        # Get another copy — should still have original data
        result2 = store.find_longest_prefix(tokens)
        self.assertIsNotNone(result2)
        _, restored_cache_2 = result2

        self.assertFalse(mx.array_equal(restored_cache_2[0]._data, mx.array([999.0])))

    def test_save_duplicate_key_is_noop(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)
        tokens = mx.array([1, 2, 3])
        cache = make_mock_cache(offset=3, data=1.0)

        store.save(tokens, cache)
        self.assertEqual(len(store), 1)

        # Save the same key again — should not add a second entry
        store.save(tokens, cache)
        self.assertEqual(len(store), 1)

    def test_checkpoint_key_longer_than_query_is_skipped(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)

        # Save a long checkpoint
        tokens_long = mx.array([1, 2, 3, 4, 5])
        store.save(tokens_long, make_mock_cache(data=5.0))

        # Query with shorter tokens — shouldn't match even though they share a prefix
        query_short = mx.array([1, 2, 3])
        result = store.find_longest_prefix(query_short)
        self.assertIsNone(result)

    def test_clear(self):
        store = RecurrentCheckpointStore(max_checkpoints=8)

        store.save(mx.array([1, 2, 3]), make_mock_cache())
        store.save(mx.array([4, 5, 6]), make_mock_cache())
        self.assertEqual(len(store), 2)

        store.clear()
        self.assertEqual(len(store), 0)

        result = store.find_longest_prefix(mx.array([1, 2, 3]))
        self.assertIsNone(result)

    def test_partial_prefix_match(self):
        """Only the first 3 tokens match — should not match a 5-token checkpoint."""
        store = RecurrentCheckpointStore(max_checkpoints=8)

        tokens = mx.array([1, 2, 3, 4, 5])
        store.save(tokens, make_mock_cache(data=5.0))

        # Query diverges at position 3
        query = mx.array([1, 2, 3, 99, 100])
        result = store.find_longest_prefix(query)
        # tokens (1,2,3,4,5) is NOT a prefix of (1,2,3,99,100), so no match
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
