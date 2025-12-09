"""
Enhanced unit tests for branching cache wrapper with comprehensive edge case coverage.

Tests LRU eviction, concurrent operations, memory pressure, and error handling
for high-bandwidth Apple Silicon support.
"""

import unittest
import threading
import time
from unittest import mock
from collections import OrderedDict

from mlx_engine.cache_wrapper import BranchingCacheWrapper, CacheSlot


class FakeCache:
    """Fake cache object for testing."""

    def __init__(self, name: str, size: int = 1024):
        self.name = name
        self.size = size
        self.data = {}

    def __len__(self):
        return len(self.data)

    def __sizeof__(self):
        return self.size


class TestBranchingCacheWrapperEnhanced(unittest.TestCase):
    """Enhanced tests for BranchingCacheWrapper with edge case coverage."""

    def test_concurrent_branch_operations(self):
        """Test thread safety of branch operations."""
        wrapper = BranchingCacheWrapper(max_slots=4)
        results = []
        errors = []

        def create_branch(branch_id):
            try:
                cache = FakeCache(f"cache_{branch_id}")
                wrapper.checkpoint_branch(branch_id, cache)
                results.append(branch_id)
            except Exception as e:
                errors.append(e)

        def restore_branch(branch_id):
            try:
                restored = wrapper.restore_branch(branch_id)
                results.append(f"restored_{branch_id}")
            except Exception as e:
                errors.append(e)

        # Create multiple threads for concurrent operations
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_branch, args=(f"branch_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have created 4 branches (max_slots)
        self.assertEqual(len(wrapper.branches), 4)
        self.assertGreaterEqual(len(results), 4)
        self.assertEqual(len(errors), 0)

    def test_memory_pressure_eviction_order(self):
        """Test eviction order under extreme memory pressure."""
        wrapper = BranchingCacheWrapper(max_slots=2, memory_headroom_ratio=0.9)

        # Mock low memory headroom
        with mock.patch.object(wrapper, "_check_memory_headroom", return_value=False):
            # Create branches to fill slots
            wrapper.checkpoint_branch("first", FakeCache("first"))
            wrapper.checkpoint_branch("second", FakeCache("second"))

            # Access first to make it most recent
            wrapper.restore_branch("first")

            # Create third branch - should evict second (oldest, not active)
            wrapper.checkpoint_branch("third", FakeCache("third"))

            # Verify eviction order
            self.assertIn("first", wrapper.branches)
            self.assertIn("third", wrapper.branches)
            self.assertNotIn("second", wrapper.branches)
            self.assertGreaterEqual(wrapper.stats.evictions, 1)

    def test_branch_metadata_tracking(self):
        """Test tracking of branch creation time, size, access patterns."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Create branches with different sizes
        wrapper.checkpoint_branch("small", FakeCache("small", 512))
        time.sleep(0.01)  # Small delay to ensure different timestamps
        wrapper.checkpoint_branch("medium", FakeCache("medium", 1024))
        time.sleep(0.01)
        wrapper.checkpoint_branch("large", FakeCache("large", 2048))

        # Access branches multiple times
        wrapper.restore_branch("small")
        wrapper.restore_branch("small")
        wrapper.restore_branch("medium")

        # Check metadata
        small_info = wrapper.get_branch_info("small")
        medium_info = wrapper.get_branch_info("medium")
        large_info = wrapper.get_branch_info("large")

        self.assertIsNotNone(small_info)
        self.assertIsNotNone(medium_info)
        self.assertIsNotNone(large_info)

        # Check access counts
        self.assertEqual(small_info["access_count"], 2)
        self.assertEqual(medium_info["access_count"], 1)
        self.assertEqual(large_info["access_count"], 0)

        # Check sizes
        self.assertEqual(small_info["size_bytes"], 512)
        self.assertEqual(medium_info["size_bytes"], 1024)
        self.assertEqual(large_info["size_bytes"], 2048)

        # Check active branch
        self.assertEqual(wrapper.active_branch_id, "medium")
        self.assertFalse(small_info["is_active"])
        self.assertTrue(medium_info["is_active"])
        self.assertFalse(large_info["is_active"])

    def test_cache_slot_limit_enforcement(self):
        """Test strict enforcement of slot limits."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Fill slots
        wrapper.checkpoint_branch("branch1", FakeCache("branch1"))
        wrapper.checkpoint_branch("branch2", FakeCache("branch2"))

        self.assertEqual(len(wrapper.branches), 2)

        # Add third branch - should evict oldest
        wrapper.checkpoint_branch("branch3", FakeCache("branch3"))

        self.assertEqual(len(wrapper.branches), 2)
        self.assertIn("branch2", wrapper.branches)  # More recent
        self.assertIn("branch3", wrapper.branches)  # Newest
        self.assertNotIn("branch1", wrapper.branches)  # Evicted

    def test_branch_restore_with_corrupted_cache(self):
        """Test error handling when cache is corrupted."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Test restoring non-existent branch
        with self.assertRaises(KeyError):
            wrapper.restore_branch("nonexistent")

        # Verify miss is recorded
        self.assertGreater(wrapper.stats.misses, 0)
        self.assertGreater(wrapper.stats.total_accesses, 0)

    def test_pinned_branch_not_evicted(self):
        """Test that pinned branches are protected from eviction."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Create pinned branch
        wrapper.checkpoint_branch("pinned", FakeCache("pinned"), pin=True)
        wrapper.checkpoint_branch("regular", FakeCache("regular"))

        # Verify pinned status
        pinned_info = wrapper.get_branch_info("pinned")
        self.assertTrue(pinned_info["pinned"])

        # Add third branch - should evict regular, not pinned
        wrapper.checkpoint_branch("new", FakeCache("new"))

        self.assertIn("pinned", wrapper.branches)
        self.assertIn("new", wrapper.branches)
        self.assertNotIn("regular", wrapper.branches)

    def test_lru_order_with_access_patterns(self):
        """Test LRU order updates with different access patterns."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Create branches
        wrapper.checkpoint_branch("a", FakeCache("a"))
        wrapper.checkpoint_branch("b", FakeCache("b"))
        wrapper.checkpoint_branch("c", FakeCache("c"))

        # Access in different order
        wrapper.restore_branch("a")  # a becomes most recent
        wrapper.restore_branch("c")  # c becomes most recent

        # Add new branch - should evict b (least recently used)
        wrapper.checkpoint_branch("d", FakeCache("d"))

        self.assertIn("a", wrapper.branches)
        self.assertIn("c", wrapper.branches)
        self.assertIn("d", wrapper.branches)
        self.assertNotIn("b", wrapper.branches)

    def test_cache_statistics_accuracy(self):
        """Test accuracy of cache statistics tracking."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Create branches
        wrapper.checkpoint_branch("branch1", FakeCache("branch1", 1024))
        wrapper.checkpoint_branch("branch2", FakeCache("branch2", 2048))

        # Perform hits and misses
        wrapper.restore_branch("branch1")  # Hit
        try:
            wrapper.restore_branch("missing")  # Miss
        except KeyError:
            pass

        # Get stats
        stats = wrapper.get_cache_stats()

        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total_accesses"], 2)
        self.assertEqual(stats["total_branches"], 2)
        self.assertEqual(stats["max_slots"], 3)
        self.assertEqual(stats["active_branch"], "branch1")
        self.assertEqual(stats["pinned_branches"], 0)
        self.assertEqual(stats["total_cache_size_bytes"], 3072)  # 1024 + 2048
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=2)
        self.assertAlmostEqual(stats["utilization"], 2 / 3, places=2)

    def test_memory_headroom_checking(self):
        """Test memory headroom checking functionality."""
        wrapper = BranchingCacheWrapper(max_slots=2, memory_headroom_ratio=0.2)

        # Test with sufficient memory
        with mock.patch.object(wrapper, "_check_memory_headroom", return_value=True):
            wrapper.checkpoint_branch("test1", FakeCache("test1"))
            wrapper.checkpoint_branch("test2", FakeCache("test2"))
            # Should allow normal operation
            self.assertEqual(len(wrapper.branches), 2)

        # Test with insufficient memory
        with mock.patch.object(wrapper, "_check_memory_headroom", return_value=False):
            wrapper.checkpoint_branch("test3", FakeCache("test3"))
            # Should trigger eviction
            self.assertGreaterEqual(wrapper.stats.evictions, 1)

    def test_branch_release_vs_eviction(self):
        """Test difference between branch release and eviction."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Create branches
        wrapper.checkpoint_branch("branch1", FakeCache("branch1"))
        wrapper.checkpoint_branch("branch2", FakeCache("branch2"))
        wrapper.checkpoint_branch("branch3", FakeCache("branch3"))

        initial_evictions = wrapper.stats.evictions

        # Release a branch (should not count as eviction)
        wrapper.release_branch("branch2")

        # Verify release doesn't increment eviction count
        self.assertEqual(wrapper.stats.evictions, initial_evictions)
        self.assertNotIn("branch2", wrapper.branches)
        self.assertIn("branch1", wrapper.branches)
        self.assertIn("branch3", wrapper.branches)

    def test_pin_unpin_operations(self):
        """Test pinning and unpinning of branches."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Create branch
        wrapper.checkpoint_branch("test_branch", FakeCache("test_branch"))

        # Pin the branch
        wrapper.pin_branch("test_branch")
        info = wrapper.get_branch_info("test_branch")
        self.assertTrue(info["pinned"])

        # Unpin the branch
        wrapper.unpin_branch("test_branch")
        info = wrapper.get_branch_info("test_branch")
        self.assertFalse(info["pinned"])

        # Test pinning non-existent branch
        with self.assertRaises(KeyError):
            wrapper.pin_branch("nonexistent")

        # Test unpinning non-existent branch
        with self.assertRaises(KeyError):
            wrapper.unpin_branch("nonexistent")

    def test_clear_cache_functionality(self):
        """Test complete cache clearing functionality."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Create branches with different states
        wrapper.checkpoint_branch("branch1", FakeCache("branch1"))
        wrapper.checkpoint_branch("branch2", FakeCache("branch2"), pin=True)
        wrapper.restore_branch("branch1")

        # Verify state before clearing
        self.assertEqual(len(wrapper.branches), 2)
        self.assertEqual(wrapper.active_branch_id, "branch1")
        self.assertGreater(wrapper.stats.total_accesses, 0)

        # Clear cache
        wrapper.clear_cache()

        # Verify state after clearing
        self.assertEqual(len(wrapper.branches), 0)
        self.assertIsNone(wrapper.active_branch_id)
        self.assertEqual(wrapper.stats.hits, 0)
        self.assertEqual(wrapper.stats.misses, 0)
        self.assertEqual(wrapper.stats.evictions, 0)
        self.assertEqual(wrapper.stats.total_accesses, 0)

    def test_list_branches_functionality(self):
        """Test listing of cached branches."""
        wrapper = BranchingCacheWrapper(max_slots=3)

        # Initially empty
        self.assertEqual(wrapper.list_branches(), [])

        # Add branches
        wrapper.checkpoint_branch("branch1", FakeCache("branch1"))
        wrapper.checkpoint_branch("branch2", FakeCache("branch2"))

        # List should contain all branches
        branches = wrapper.list_branches()
        self.assertEqual(len(branches), 2)
        self.assertIn("branch1", branches)
        self.assertIn("branch2", branches)

    def test_edge_case_slot_configurations(self):
        """Test edge cases for slot configurations."""
        # Test minimum slots
        wrapper1 = BranchingCacheWrapper(max_slots=1)
        self.assertEqual(wrapper1.max_slots, 1)

        # Test invalid configurations
        with self.assertRaises(ValueError):
            BranchingCacheWrapper(max_slots=0)

        with self.assertRaises(ValueError):
            BranchingCacheWrapper(max_slots=-1)

        with self.assertRaises(ValueError):
            BranchingCacheWrapper(eviction_policy="invalid")

        with self.assertRaises(ValueError):
            BranchingCacheWrapper(memory_headroom_ratio=-0.1)

        with self.assertRaises(ValueError):
            BranchingCacheWrapper(memory_headroom_ratio=1.1)

    def test_cache_size_estimation(self):
        """Test cache size estimation for different cache types."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Test with cache that has __sizeof__
        cache_with_size = FakeCache("test", 4096)
        size = wrapper._estimate_cache_size(cache_with_size)
        self.assertEqual(size, 4096)

        # Test with cache that has __len__
        cache_with_len = FakeCache("test")
        cache_with_len.__sizeof__ = None  # Remove __sizeof__
        size = wrapper._estimate_cache_size(cache_with_len)
        self.assertEqual(size, 1024)  # len * 1024

        # Test with minimal cache (fallback)
        class MinimalCache:
            pass

        minimal_cache = MinimalCache()
        minimal_cache.__sizeof__ = None  # Remove __sizeof__ to test fallback
        size = wrapper._estimate_cache_size(minimal_cache)
        self.assertEqual(size, 1024)  # 1KB fallback

    def test_force_eviction_scenarios(self):
        """Test force eviction when normal eviction fails."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Create pinned branch and regular branch
        wrapper.checkpoint_branch("pinned", FakeCache("pinned"), pin=True)
        wrapper.checkpoint_branch("regular", FakeCache("regular"))

        # Make regular branch active
        wrapper.restore_branch("regular")

        # Mock memory headroom check to always return False
        with mock.patch.object(wrapper, "_check_memory_headroom", return_value=False):
            # Try to add new branch - should force eviction of regular
            # (can't evict pinned, even though regular is active)
            wrapper.checkpoint_branch("new", FakeCache("new"))

            # Should have evicted regular despite being active
            self.assertIn("pinned", wrapper.branches)
            self.assertIn("new", wrapper.branches)
            self.assertNotIn("regular", wrapper.branches)

    def test_branch_info_edge_cases(self):
        """Test branch info retrieval for edge cases."""
        wrapper = BranchingCacheWrapper(max_slots=2)

        # Test info for non-existent branch
        info = wrapper.get_branch_info("nonexistent")
        self.assertIsNone(info)

        # Create branch and test info
        wrapper.checkpoint_branch("test", FakeCache("test"))
        info = wrapper.get_branch_info("test")

        # Verify all expected fields
        expected_fields = [
            "branch_id",
            "last_used",
            "size_bytes",
            "pinned",
            "access_count",
            "is_active",
        ]
        for field in expected_fields:
            self.assertIn(field, info)

        # Verify field types
        self.assertIsInstance(info["branch_id"], str)
        self.assertIsInstance(info["last_used"], float)
        self.assertIsInstance(info["size_bytes"], int)
        self.assertIsInstance(info["pinned"], bool)
        self.assertIsInstance(info["access_count"], int)
        self.assertIsInstance(info["is_active"], bool)


if __name__ == "__main__":
    unittest.main()
