import unittest

from mlx_engine.cache_wrapper import BranchingCacheWrapper


class TestCacheMetricsShape(unittest.TestCase):
    def test_stats_keys_present_and_ints(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        stats = wrapper.stats
        for key in ["hits", "misses", "evictions"]:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], int)
        self.assertGreaterEqual(stats.get("hits", 0), 0)
        self.assertGreaterEqual(stats.get("misses", 0), 0)
        self.assertGreaterEqual(stats.get("evictions", 0), 0)


if __name__ == "__main__":
    unittest.main()
