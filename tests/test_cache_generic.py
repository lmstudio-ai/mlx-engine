import unittest
import mlx.core as mx
import mlx.nn as nn


class TestCache(unittest.TestCase):
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