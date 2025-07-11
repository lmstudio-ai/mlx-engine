import unittest
import mlx.core as mx
from mlx_engine.cache import ShiftingKVCache
from tests.test_cache_generic import TestCache


def idx(v: mx.array, i: int):
    """Helper function to index into a 4D tensor at the sequence length dimension"""
    return v[:, :, i : i + 1, :]


class TestShiftingKVCache(TestCache):
    def test_overwriting(self):
        """Test overwriting when the cache reaches max_size"""
        cache = ShiftingKVCache(max_size=3, keep=1)

        # fill cache -> 123
        reference = self.add_random_to_cache(cache, 3)
        self.assertEqual(cache.offset, 3)

        # attempt to write another element 4 -> 143
        overwrite = self.add_random_to_cache(cache, 1)
        # access k/v as cache.state[0]/[1] due to possibly empty buffer
        keys = cache.state[0]

        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), idx(reference, 2))
        self.assertEqual(cache.offset, 4)

    def test_ensure_update_increases_offset_indefinitely(self):
        """Test single-token updates that should increase offset"""
        cache = ShiftingKVCache(max_size=3, keep=1)

        for i in range(10):
            self.add_random_to_cache(cache, 1)
            self.assertEqual(cache.offset - 1, i)

    def test_ensure_reasonable_size_and_shift(self):
        """Test behavior when the cache gets a KV batch-written that is much larger
        than max_size. The default behavior of the cache is to write the entire thing,
        then trim it back down when the next KV is written.
        """
        cache = ShiftingKVCache(max_size=3, keep=1)

        # fill cache -> 0123456789
        reference = self.add_random_to_cache(cache, 10)
        keys = cache.state[0]
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 10, self.kv_head_dim))
        self.assertEqual(cache.offset, 10)

        # trigger trim -> 0X9 -> (rope) 021
        overwrite = self.add_random_to_cache(cache, 1)
        keys = cache.state[0]
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 3, self.kv_head_dim))
        self.assertEqual(cache.offset, 11)

        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), idx(reference, 9))

        # make sure pos embs are right
        cache.keys = cache._temporal_order(cache.keys)
        cache.values = cache._temporal_order(cache.values)
        keys = cache.state[0]

        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertArrEqual(idx(keys, 1), idx(reference, 9))
        self.assertArrEqual(idx(keys, 2), overwrite)
        self.assertEqual(cache.offset, 11)

        # ensure offset keeps increasing
        self.add_random_to_cache(cache, 1)
        self.assertEqual(cache.offset, 12)

        self.add_random_to_cache(cache, 1)
        self.assertEqual(cache.offset, 13)

    def test_update_keep_on_the_fly(self):
        """Test changing the keep value on the fly"""
        cache = ShiftingKVCache(max_size=4, keep=1)

        # fill cache -> 1234
        reference = self.add_random_to_cache(cache, 4)

        # attempt to write another element 5 -> 1534
        overwrite = self.add_random_to_cache(cache, 1)
        self.assertEqual(cache.offset, 5)

        # update keep -> 1345 -> 1234 implicitly
        # and attempt to write another element 5 -> 1254
        # offset updates after set_keep (anytime we reorder/rope shift)
        cache.set_keep(2)
        self.assertEqual(cache.offset, 5)
        overwrite2 = self.add_random_to_cache(cache, 1)
        self.assertEqual(cache.offset, 6)
        keys = cache.state[0]

        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertArrEqual(idx(keys, 1), idx(reference, 2))
        self.assertArrEqual(idx(keys, 2), overwrite2)
        self.assertArrEqual(idx(keys, 3), overwrite)

    def test_trim_before_full(self):
        """Test trimming from the end before the cache is full"""
        cache = ShiftingKVCache(max_size=3, keep=1)

        # fill cache -> 12
        reference = self.add_random_to_cache(cache, 2)

        # trim 1 from end -> 1
        cache.trim(1)
        keys = cache.state[0]

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 1, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertEqual(cache.offset, 1)

        # ensure adding another value works fine
        new_kv = self.add_random_to_cache(cache, 1)
        keys = cache.state[0]
        self.assertEqual(cache.offset, 2)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(reference, 0))
        self.assertArrEqual(idx(keys, 1), new_kv)
        self.assertEqual(cache.offset, 2)

    def test_trim_after_overwrite(self):
        """Test trimming from the end when we've written past the cache max"""
        cache = ShiftingKVCache(max_size=3, keep=1)

        # fill cache -> 123
        reference = self.add_random_to_cache(cache, 3)
        self.assertEqual(cache.offset, 3)

        # overwrite so offset goes over max_size -> 143
        base_kv = self.make_random_kv(1)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 4)

        # trim 1 from end -> 13 -> 12 (rope), ideally
        cache.trim(1)
        keys = cache.state[0]

        should_be_kv = mx.concatenate(
            [reference[:, :, :1, :], reference[:, :, 2:3, :]], axis=2
        )
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(keys, should_be_kv)
        self.assertEqual(cache.offset, 2)

    def test_trim_after_full(self):
        """Test trimming from the end when the cache is oversize"""
        cache = ShiftingKVCache(max_size=3, keep=1)

        # fill cache oversize already -> 1234
        reference = self.add_random_to_cache(cache, 4)
        self.assertEqual(cache.offset, 4)

        # trim 2 from end -> 12
        cache.trim(2)
        keys = cache.state[0]

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(keys, reference[:, :, :2, :])
        self.assertEqual(cache.offset, 2)

        # ensure adding more values works fine
        new_kv = self.add_random_to_cache(cache, 2)
        keys = cache.state[0]
        self.assertEqual(cache.offset, 4)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 4, self.kv_head_dim))
        self.assertArrEqual(keys[:, :, :2, :], reference[:, :, :2, :])
        self.assertArrEqual(keys[:, :, 2:, :], new_kv)

    def test_reuse(self):
        """Test basic reuse APIs"""
        cache = ShiftingKVCache(max_size=8, keep=1)

        # fill cache -> 12345678
        reference = self.add_random_to_cache(cache, 8)

        # reuse a specific section (hardcoded), dynamic reuse is in test_cache_wrapper
        cache.reuse_section(3, 4, 2)
        cache.do_reuse()
        keys = cache.state[0]

        # this is what the remaining cache should look like
        should_be_keys = mx.concatenate(
            [reference[:, :, :3, :], reference[:, :, 4:6, :]], axis=2
        )

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 5, self.kv_head_dim))
        self.assertArrEqual(keys, should_be_keys)
        self.assertEqual(cache.offset, 5)

    def test_reuse_after_overwrite(self):
        """Test basic reuse APIs after an overwrite"""
        cache = ShiftingKVCache(max_size=8, keep=1)

        # fill cache -> 12345678
        reference = self.add_random_to_cache(cache, 8)
        news = self.add_random_to_cache(cache, 1)  # overwrite to 13456789 after TO
        self.assertArrEqual(
            cache.state[0], mx.concatenate(
                [reference[:, :, :1, :], news, reference[:, :, 2:8, :]], axis=2
            )
        )

        # suppose the prompt coming in is now 13678
        # reuse from 2 to 4 length 3
        cache.reuse_section(2, 4, 3)
        cache.do_reuse()
        keys = cache.state[0]

        # the remaining cache should be 13678
        should_be_keys = mx.concatenate(
            [reference[:, :, :1, :], reference[:,:,2:3,:], reference[:, :, 5:8, :]], axis=2
        )

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 5, self.kv_head_dim))
        self.assertArrEqual(keys, should_be_keys)
        self.assertEqual(cache.offset, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
