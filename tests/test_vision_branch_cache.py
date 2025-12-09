import unittest

from mlx_engine.cache_wrapper import BranchingCacheWrapper


class FakeVisionCache:
    def __init__(self, name: str, media=None):
        self.name = name
        self.media = media or []


class TestVisionBranchCache(unittest.TestCase):
    def test_branch_restore_with_vision_cache(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        vision_cache = FakeVisionCache("vision")
        wrapper.checkpoint_branch("vision", vision_cache)
        restored = wrapper.restore_branch("vision")
        self.assertIs(restored, vision_cache)
        self.assertEqual(wrapper.active_branch_id, "vision")

    def test_eviction_keeps_active_vision_branch(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("vision", FakeVisionCache("vision"))
        wrapper.checkpoint_branch("text", FakeVisionCache("text"))
        wrapper.restore_branch("vision")
        wrapper.checkpoint_branch("third", FakeVisionCache("third"))

        self.assertIn("vision", wrapper.branches)
        self.assertIn("third", wrapper.branches)

    def test_eviction_counts_stats_with_mixed_cache_types(self):
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("vision", FakeVisionCache("vision"))
        wrapper.checkpoint_branch("text", FakeVisionCache("text"))
        wrapper.restore_branch("vision")
        evictions_before = wrapper.stats.get("evictions", 0)

        wrapper.checkpoint_branch("third", FakeVisionCache("third"))

        self.assertIn("vision", wrapper.branches)
        self.assertIn("third", wrapper.branches)
        self.assertGreaterEqual(wrapper.stats.get("evictions", 0), evictions_before + 1)


if __name__ == "__main__":
    unittest.main()
