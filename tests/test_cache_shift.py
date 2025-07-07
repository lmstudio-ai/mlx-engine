import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_engine.cache import ShiftingKVCache

def idx(v: mx.array, i: int):
    return v[:, :, i, :]


class ShiftingCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that will be shared across all test methods"""
        cls.rope = nn.RoPE(dims=8, traditional=False, base=10000, scale=1.0)
        cls.bsz = 1
        cls.n_kv_heads = 1
        cls.kv_head_dim = 4
        
    def make_random_kv(self, seqlen: int):
        """Helper method to make a random key/value tensor of the right shape"""
        return mx.random.normal((self.bsz, self.n_kv_heads, seqlen, self.kv_head_dim), dtype=mx.float16)
        
    # TODO: you can test to make sure that it's RoPEing right in the model overall by getting
    # the post-shift value, then shifting it back to position 0 and checking the layer 0 kv
    # matches the raw token embedding 
        
    def test_overwriting(self):
        cache = ShiftingKVCache(self.rope, max_size=3, keep=0)
        base_kv = self.make_random_kv(3)
        overwrite = self.make_random_kv(1)
        overwrite_posemb_4 = self.rope(overwrite, 4)
        cache.update_and_fetch(base_kv, base_kv)
        keys, _ = cache.update_and_fetch(overwrite, overwrite)
        self.assertEqual(overwrite_posemb_4, keys[:, :, 0, :])
        
    def test_temporal_order_shift(self):
        cache = ShiftingKVCache(self.rope, max_size=3, keep=0)
        base_kv = self.make_random_kv(3)
        overwrite = self.make_random_kv(1)
        overwrite_posemb_3 = self.rope(overwrite, 3)
        cache.update_and_fetch(base_kv, base_kv)
        cache.update_and_fetch(overwrite, overwrite)
        cache.keys = cache._temporal_order(cache.keys)
        self.assertEqual(overwrite_posemb_3, cache.keys)
        
    def test_trim_internal(self):
        pass

    def test_trim_before_full(self):
        pass
    
    def test_trim_after_full(self):
        pass
    
    def test_reuse(self):
        pass