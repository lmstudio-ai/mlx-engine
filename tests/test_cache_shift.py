import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_engine.cache import ShiftingKVCache


def idx(v: mx.array, i: int):
    """Helper function to index into a 4D tensor at the sequence length dimension"""
    return v[:, :, i : i + 1, :]


class ShiftingCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.kv_head_dim = 4
        cls.bsz = 1
        cls.n_kv_heads = 1
        # cannot be used raw: must be wrapped in the cache.rope workaround impl
        cls._rope = nn.RoPE(
            dims=cls.kv_head_dim, traditional=False, base=100000, scale=1.0
        )

    @classmethod
    def make_random_kv(cls, seqlen: int):
        """Helper method to make a random key/value tensor of the right shape"""
        return mx.random.normal(
            (cls.bsz, cls.n_kv_heads, seqlen, cls.kv_head_dim),
            scale=1.0,
            dtype=mx.float32,
        )

    def assertArrEqual(self, a: mx.array, b: mx.array):
        """Assert that two tensors are equal over the sequence length dimension"""
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(mx.allclose(a, b), "Tensors are not equal")

    # TODO: you can test to make sure that it's RoPEing right in the model overall by getting
    # the post-shift value, then shifting it back to position 0 and checking the layer 0 kv
    # matches the raw token embedding

    def test_overwriting(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(3)
        overwrite = self.make_random_kv(1)
        cache.update_and_fetch(base_kv, base_kv)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), idx(base_kv, 2))

    def test_temporal_order_shift(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(3)
        overwrite = self.make_random_kv(1)
        overwrite_roped = cache.rope(overwrite, -1)
        cache.update_and_fetch(base_kv, base_kv)
        cache.update_and_fetch(overwrite, overwrite)
        print(base_kv)
        print(cache.keys)
        keys = cache._temporal_order(cache.keys)

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))
        self.assertArrEqual(idx(keys, 2), overwrite_roped)

    def test_trim_internal_shift(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(3)
        cache.update_and_fetch(base_kv, base_kv)

        keys = cache._trim(1, cache.keys)

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), cache.rope(idx(base_kv, 2), -1))

    def test_ensure_reasonable_size_and_shift(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(10)
        keys, _ = cache.update_and_fetch(base_kv, base_kv)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 10, self.kv_head_dim))
        overwrite = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 3, self.kv_head_dim))

        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), overwrite)
        self.assertArrEqual(idx(keys, 2), cache.rope(idx(base_kv, 9), -7))

    def test_trim_before_full(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(2)
        cache.update_and_fetch(base_kv, base_kv)

        cache.trim(1)
        keys = cache.keys

        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 1, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))

        new_kv = self.make_random_kv(1)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(idx(keys, 0), idx(base_kv, 0))
        self.assertArrEqual(idx(keys, 1), new_kv)

    def test_trim_after_full(self):
        cache = ShiftingKVCache(self._rope, max_size=3, keep=1)
        base_kv = self.make_random_kv(4)
        cache.update_and_fetch(base_kv, base_kv)

        cache.trim(2)
        keys = cache.keys
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 2, self.kv_head_dim))
        self.assertArrEqual(keys, base_kv[:, :, :2, :])

        new_kv = self.make_random_kv(2)
        keys, _ = cache.update_and_fetch(new_kv, new_kv)
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 4, self.kv_head_dim))
        self.assertArrEqual(keys[:, :, :2, :], base_kv[:, :, :2, :])
        self.assertArrEqual(keys[:, :, 2:, :], new_kv)

    def test_reuse(self):
        cache = ShiftingKVCache(self._rope, max_size=6, keep=1)
        base_kv = self.make_random_kv(8)
        cache.update_and_fetch(base_kv, base_kv)
        new_prompt_cache = mx.concatenate(
            [base_kv[:, :, :3, :], cache.rope(base_kv[:, :, 4:, :], -1)], axis=2
        )
        # here we know what to reuse so hardcode it, dynamic reuse is in test_cache_wrapper
        cache.reuse_section(3, 4, 2)
        cache.do_reuse()
        keys = cache.keys
        self.assertEqual(keys.shape, (self.bsz, self.n_kv_heads, 5, self.kv_head_dim))
        self.assertArrEqual(keys, new_prompt_cache[:, :, :5, :])
