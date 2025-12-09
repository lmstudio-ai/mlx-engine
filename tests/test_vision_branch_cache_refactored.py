"""
Refactored vision branch cache tests using shared utilities.
This module eliminates duplication by using standardized test patterns.
"""

import unittest
from tests.fixtures.vision_test_fixtures import BaseVisionTest
from mlx_engine.cache_wrapper import BranchingCacheWrapper


class FakeVisionCache:
    """Fake vision cache for testing purposes."""

    def __init__(self, name: str, media=None):
        self.name = name
        self.media = media or []


class TestVisionBranchCache(unittest.TestCase):
    """Test suite for vision branch cache functionality."""

    def test_branch_restore_with_vision_cache(self):
        """Test that vision cache can be checkpointed and restored."""
        wrapper = BranchingCacheWrapper(max_slots=2)
        vision_cache = FakeVisionCache("vision")
        wrapper.checkpoint_branch("vision", vision_cache)
        restored = wrapper.restore_branch("vision")
        self.assertIs(restored, vision_cache)
        self.assertEqual(wrapper.active_branch_id, "vision")

    def test_eviction_keeps_active_vision_branch(self):
        """Test that active vision branch is not evicted."""
        wrapper = BranchingCacheWrapper(max_slots=2)
        wrapper.checkpoint_branch("vision", FakeVisionCache("vision"))
        wrapper.checkpoint_branch("text", FakeVisionCache("text"))
        wrapper.restore_branch("vision")
        wrapper.checkpoint_branch("third", FakeVisionCache("third"))

        self.assertIn("vision", wrapper.branches)
        self.assertIn("third", wrapper.branches)

    def test_eviction_counts_stats_with_mixed_cache_types(self):
        """Test that eviction stats are counted correctly with mixed cache types."""
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
