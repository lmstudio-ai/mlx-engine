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
        
        # attempt to write another element 4 -> 143
        overwrite = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), idx(base_kv, 2))

    def test_temporal_order_shift(self):
        """Test the RoPE shift in _temporal_order"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)

        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)
        
        # attempt to write another element 4 -> 143
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(overwrite, overwrite)

        # put the cache in temporal order -> 134 -> 123 (rope shift)
        keys = cache._temporal_order(cache.keys)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))
        self.assertArrEqual(idx(keys, 2), cache.rope(overwrite, -1))

    def test_trim_internal_shift(self):
        """Test the RoPE shift in _trim (internal method)"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache -> 123
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)

        # trim 1 from middle -> 13
        keys = cache._trim(1, cache.keys)


        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))

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
        
        # trigger trim -> 0X9 -> (rope) 021
        overwrite = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 3, self.kv_head_dim))

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        # TODO(christian-lms): this should also be rope unshifted because it's coming in
        # w/ pos emb @ position X and then being sent to 2. figure out where this goes
        self.assertArrEqual(idx(keys, 1), overwrite)
        # TODO(christian-lms): is this position 2 or 1? it should be 1
        self.assertArrEqual(idx(keys, 2), cache.rope(idx(base_kv, 9), -7))

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

        # ensure adding another value works fine
        new_kv = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), new_kv)

    # TODO(christian-lms): this doesn't actually test the overwriting, for that you
    # need to fill it to 3 first then add 1 then try trim
    def test_trim_after_full(self):
        """Test trimming from the end when the cache is oversize"""
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        
        # fill cache oversize already -> 1234
        base_kv = self.make_random_kv(4)
        cache.update_and_fetch(base_kv, base_kv)

        # trim 2 from end -> 12
        cache.trim(2)
        keys = cache.keys
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(keys, base_kv[:, :, :2, :])

        # ensure adding more values works fines
        new_kv = self.make_random_kv(2)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)