import unittest
import mlx.core as mx
from copy import deepcopy
from mlx_engine.cache import AlwaysTrimmableKVCache


class TestCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.kv_head_dim = 4
        cls.bsz = 1
        cls.n_kv_heads = 1

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

    def add_random_to_cache(
        self, cache: AlwaysTrimmableKVCache, seqlen: int
    ) -> mx.array:
        """Add random values to the cache and return them"""
        base_kv = self.make_random_kv(seqlen)
        # base_kv is *assigned* to cache.keys/cache.values so returning base_kv
        # would return a reference to cache.keys, which is pointless. so copy it
        reference = deepcopy(base_kv)
        cache.update_and_fetch(base_kv, base_kv)
        return reference
