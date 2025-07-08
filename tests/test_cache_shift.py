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
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)

        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 3)
        
        # attempt to write another element 4 -> 143
        overwrite = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), idx(base_kv, 2))
        self.assertEqual(cache.offset, 4)
        
    def test_ensure_update_increases_offset_indefinitely(self):
        """Test single-token updates that should increase offset"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        for i in range(10):
            kv = self.make_random_kv(1)
            cache.update_and_fetch(kv, kv)
            self.assertEqual(cache.offset - 1, i)

    def test_temporal_order_shift_rope(self):
        """Test the RoPE shift in _temporal_order"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)

        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 3)
        
        # attempt to write another element 4 -> 143
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(cache.offset, 4)

        # put the cache in temporal order -> 134 -> 123 (rope shift)
        cache._temporal_order()
        keys = cache.keys

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))
        self.assertArrEqual(idx(keys, 2), cache.rope(overwrite, -1))
        self.assertEqual(cache.offset, 3)

    def test_temporal_order_shift_no_rope(self):
        """Test putting the cache in temporal order"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)

        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 3)
        
        # attempt to write another element 4 -> 143
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(cache.offset, 4)

        # put the cache in temporal order -> 134 (no rope shift)
        cache._temporal_order()
        values = cache.values

        self.assertArrEqual(idx(values, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(values, 1), idx(base_kv, 2))
        self.assertArrEqual(idx(values, 2), overwrite)
        self.assertEqual(cache.offset, 3)

    def test_trim_internal_shift_rope(self):
        """Test the RoPE shift in _trim (internal method)"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 3)

        # trim 1 from middle -> 13
        cache._trim(1)
        keys = cache.keys

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))
        # trim should trigger offset change with is_key=True
        self.assertEqual(cache.offset, 2)

    def test_trim_internal_shift_no_rope(self):
        """Test the RoPE shift in _trim (internal method)"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 3)

        # trim 1 from middle -> 13 -> 12
        cache._trim(1)
        values = cache.values
        
        self.assertEqual(values.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(values, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(values, 1), idx(base_kv, 2))
        self.assertEqual(cache.offset, 2)

    def test_ensure_reasonable_size_and_shift(self):
        """Test behavior when the cache gets a KV batch-written that is much larger
        than max_size. The default behavior of the cache is to write the entire thing,
        then trim it back down when the next KV is written.
        """
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache -> 0123456789
        base_kv = self.make_random_kv(10)
        keys, _ = cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 10, self.kv_head_dim))
        self.assertEqual(cache.offset, 10)
        
        # trigger trim -> 0X9 -> (rope) 021
        overwrite = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 3, self.kv_head_dim))
        # this should be 4 since this mimics autoregression
        self.assertEqual(cache.offset, 4)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), cache.rope(idx(base_kv, 9), -7))

        # make sure pos embs are right
        cache._temporal_order()
        keys = cache.keys
        
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 9), -8))
        self.assertArrEqual(idx(keys, 2), cache.rope(overwrite, -1))
        self.assertEqual(cache.offset, 3)
        
        # ensure offset keeps increasing
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(cache.offset, 4)

        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(cache.offset, 5)
    
    def test_update_keep_on_the_fly(self):
        """Test changing the keep value on the fly"""
        cache = ShiftingKVCache(self._rope, max_size=4, keep=1)

        # fill cache -> 1234
        base_kv = self.make_random_kv(4)
        cache.update_and_fetch(base_kv, base_kv)

        # attempt to write another element 5 -> 1534
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(cache.offset, 5)

        # update keep -> 1345 -> 1234 implicitly
        # and attempt to write another element 5 -> 1254
        # offset updates after set_keep (anytime we reorder/rope shift) 
        cache.set_keep(2)
        overwrite2 = self.make_random_kv(1)
        self.assertEqual(cache.offset, 4)
        keys, _ = cache.update_and_fetch(overwrite2, overwrite2)
        self.assertEqual(cache.offset, 5)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))
        self.assertArrEqual(idx(keys, 2), overwrite2)
        self.assertArrEqual(idx(keys, 3), cache.rope(overwrite, -1))
        
    # TODO add offset assertions everywhere to make sure you're good
        
    def test_trim_before_full(self):
        """Test trimming from the end before the cache is full"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache -> 12
        base_kv = self.make_random_kv(2)
        cache.update_and_fetch(base_kv, base_kv)

        # trim 1 from end -> 1
        cache.trim(1)
        keys = cache.keys

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 1, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertEqual(cache.offset, 1)

        # ensure adding another value works fine
        new_kv = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), new_kv)
        self.assertEqual(cache.offset, 2)

    # TODO(christian-lms): this doesn't actually test the overwriting, for that you
    # need to fill it to 3 first then add 1 then try trim
    def test_trim_after_full(self):
        """Test trimming from the end when the cache is oversize"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache oversize already -> 1234
        base_kv = self.make_random_kv(4)
        cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(cache.offset, 4)

        # trim 2 from end -> 12
        cache.trim(2)
        keys = cache.keys
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(keys, base_kv[:, :, :2, :])
        self.assertEqual(cache.offset, 2)

        # ensure adding more values works fine
        new_kv = self.make_random_kv(2)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)
        self.assertEqual(cache.offset, 4)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 4, self.kv_head_dim))
        self.assertArrEqual(keys[:, :, :2, :], base_kv[:, :, :2, :])
        self.assertArrEqual(keys[:, :, 2:, :], new_kv)

    def test_reuse(self):
        """Test basic reuse APIs"""
        cache = ShiftingKVCache(self._rope, max_size=8, keep=1)
        
        # fill cache -> 12345678
        base_kv = self.make_random_kv(8)
        cache.update_and_fetch(base_kv, base_kv)
        
        # reuse a specific section (hardcoded), dynamic reuse is in test_cache_wrapper
        cache.reuse_section(3, 4, 2)
        cache.do_reuse()
        keys = cache.keys

        # this is what the remaining cache should look like
        should_be_keys = mx.concatenate(
            [base_kv[:, :, :3, :], cache.rope(base_kv[:, :, 4:6, :], -1)], axis=2
        )

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 5, self.kv_head_dim))
        self.assertArrEqual(keys, should_be_keys)
        self.assertEqual(cache.offset, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)