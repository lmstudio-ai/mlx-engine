import unittest

from mlx_engine.cache_wrapper import BranchingCacheWrapper


class FakeCache:
    def __init__(self, name: str):
        self.name = name


class TestBranchingCacheWrapper(unittest.TestCase):
    def test_lru_eviction_skips_active_branch(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("a", FakeCache("a"))
        wrapper.checkpoint_branch("b", FakeCache("b"))
        wrapper.restore_branch("a")  # mark a as most recent/active

        wrapper.checkpoint_branch("c", FakeCache("c"))  # should evict b

        self.assertEqual(wrapper.active_branch_id, "a")
        self.assertIn("a", wrapper.branches)
        self.assertIn("c", wrapper.branches)
        self.assertNotIn("b", wrapper.branches)
        self.assertGreaterEqual(wrapper.stats.get("evictions", 0), 1)

    def test_restore_missing_branch_raises(self):
        wrapper = BranchingCacheWrapper(max_slots=1)
        with self.assertRaises(KeyError):
            wrapper.restore_branch("missing")

    def test_pinned_branch_not_evicted(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("root", FakeCache("root"), pin=True)
        wrapper.checkpoint_branch("side", FakeCache("side"))
        wrapper.checkpoint_branch("third", FakeCache("third"))

        self.assertIn("root", wrapper.branches)
        self.assertIn("third", wrapper.branches)
        self.assertNotIn("side", wrapper.branches)
        self.assertTrue(wrapper.branches["root"].pinned)

    def test_stats_record_hits_and_misses(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("root", FakeCache("root"))
        wrapper.checkpoint_branch("alt", FakeCache("alt"))

        wrapper.restore_branch("root")
        with self.assertRaises(KeyError):
            wrapper.restore_branch("missing")

        hits = wrapper.stats.get("hits", 0)
        misses = wrapper.stats.get("misses", 0)
        self.assertGreaterEqual(hits, 1)
        self.assertGreaterEqual(misses, 1)

    def test_release_branch_drops_slot_without_eviction(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("root", FakeCache("root"))
        evictions_before = wrapper.stats.get("evictions", 0)

        wrapper.release_branch("root")

        self.assertNotIn("root", wrapper.branches)
        self.assertEqual(wrapper.stats.get("evictions", 0), evictions_before)

    def test_restore_returns_cached_object(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        cache = FakeCache("root")
        wrapper.checkpoint_branch("root", cache)
        restored = wrapper.restore_branch("root")
        self.assertIs(restored, cache)
        self.assertEqual(wrapper.active_branch_id, "root")


if __name__ == "__main__":
    unittest.main()
