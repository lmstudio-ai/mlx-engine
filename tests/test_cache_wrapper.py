import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper
from mlx_engine.cache import ShiftingKVCache
from tests.test_cache_generic import TestCache
from tests.utils import DummyModel, model_getter
from mlx_engine.generate import load_model, create_generator


class TestCacheWrapper(TestCache):
    def test_find_matching_sequence_length_with_mismatch(self):
        """Test when there's a mismatch in the tokens"""
        # Create two arrays with a known common prefix [1, 2, 3]
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 6, 7])  # Mismatch at index 3

        print("\nTest with mismatch:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens
        )
        self.assertEqual(result, 3)  # Should find 3 matching tokens

    def test_find_matching_sequence_length_all_match(self):
        """Test when all tokens match"""
        # Create two identical arrays
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])  # All tokens match

        print("\nTest with all matching:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens
        )
        self.assertEqual(result, 5)  # Should find 5 matching tokens

    def test_find_matching_sequence_length_no_match(self):
        """Test when no tokens match"""
        # Create two arrays with no common prefix
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([6, 7, 8, 9, 10])

        print("\nTest with no matching tokens:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens
        )
        self.assertEqual(result, 0)  # No matching tokens should return 0

    def test_find_matching_sequence_length_offset_starts(self):
        """Test when the current tokens start with a different offset"""
        # Create two arrays where the current tokens start with a different offset
        current_tokens = mx.array([2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])

        print("\nTest with offset starts:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens,
            prompt_tokens,
            start2=1,
        )
        self.assertEqual(result, 4)

    def test_find_matching_sequence_length_more_offsets(self):
        """Test when the current tokens have more offsets"""
        # Create two arrays where the current tokens have more offsets
        current_tokens = mx.array([1, 2, 3, 4, 5, 6])
        prompt_tokens = mx.array([0, 9, 10, 3, 4, 7, 8])

        print("\nTest with more offsets:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens, prompt_tokens
        )
        self.assertEqual(result, 0)

        result = CacheWrapper._find_matching_sequence_length(
            current_tokens,
            prompt_tokens,
            start1=2,
            start2=3,
        )
        self.assertEqual(result, 2)

    def test_record_generated_token_loops(self):
        cache = CacheWrapper(
            model=DummyModel(),
            max_kv_size=5,
            keep=2,
        )
        cache.tokens = mx.array([])
        cache.record_generated_token(1)
        cache.record_generated_token(2)
        cache.record_generated_token(3)
        cache.record_generated_token(4)
        cache.record_generated_token(5)
        self.assertListEqual(
            cache.tokens.tolist(),
            [1, 2, 3, 4, 5],
        )
        cache.record_generated_token(6)
        self.assertListEqual(
            cache.tokens.tolist(),
            [1, 2, 4, 5, 6],
        )

    def test_cache_reuse_heavy(self):
        cache = CacheWrapper(DummyModel(), 10, keep=2)
        cache.cache[0] = ShiftingKVCache(max_size=10, keep=2)

        # set up pretend cache
        cached_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cache_kv = self.make_random_kv(10)
        cache.tokens = cached_tokens
        cache.cache[0].update_and_fetch(cache_kv, cache_kv)

        # set up pretend prompt
        prompt_tokens = mx.array([1, 2, 4, 7, 8, 9, 11])

        prefix_len = cache._find_matching_sequence_length(
            cached_tokens, prompt_tokens, 0
        )
        self.assertEqual(prefix_len, 2)

        total_reused = cache._truncate_cache(
            prompt_tokens=prompt_tokens,
            common_prefix_len=prefix_len,
            non_prefix_reuse_min_seq_len=1,
        )

        # prepare references
        def idx(v, a, b):
            return v[:, :, a:b, :]

        should_be_tokens = mx.array([1, 2, 4, 7, 8, 9])
        should_be_kv = mx.concatenate(
            [
                idx(cache_kv, 0, 2),
                idx(cache_kv, 3, 4),
                idx(cache_kv, 6, 9),
            ],
            axis=2,
        )

        self.assertEqual(total_reused, 4)
        self.assertArrEqual(cache.tokens, should_be_tokens)
        self.assertArrEqual(cache.cache[0].keys, should_be_kv)

        # ensure updating works as intended
        new_kv = self.make_random_kv(1)
        keys, _ = cache.cache[0].update_and_fetch(new_kv, new_kv)
        should_be_kv = mx.concatenate([should_be_kv, new_kv], axis=2)
        self.assertArrEqual(keys, should_be_kv)

        # ensure batch concat works as intended
        new_kv = self.make_random_kv(2)
        keys, _ = cache.cache[0].update_and_fetch(new_kv, new_kv)
        should_be_kv = mx.concatenate([should_be_kv, new_kv], axis=2)
        self.assertArrEqual(keys, should_be_kv)

    def test_cache_reuse_overwrite_heavy(self):
        cache = CacheWrapper(DummyModel(), 10, keep=2)
        cache.cache[0] = ShiftingKVCache(max_size=10, keep=2)

        # set up pretend cache
        cached_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cache_kv = self.make_random_kv(10)
        for i in range(10):
            cache.record_generated_token(cached_tokens[i])
        cache.cache[0].update_and_fetch(cache_kv, cache_kv)

        # append another one to overwrite
        cache.record_generated_token(11)
        cache_new_kv = self.make_random_kv(1)
        cache.cache[0].update_and_fetch(cache_new_kv, cache_new_kv)

        print(cache.tokens)
        self.assertArrEqual(cache.tokens, mx.array([1, 2, 4, 5, 6, 7, 8, 9, 10, 11]))
        self.assertEqual(cache.cache[0].keys.shape[2], 10)

        # set up pretend prompt
        prompt_tokens = mx.array([1, 2, 4, 7, 8, 9, 12])

        prefix_len = cache._find_matching_sequence_length(
            cached_tokens, prompt_tokens, 0
        )
        self.assertEqual(prefix_len, 2)

        # prepare references
        def idx(v, a, b):
            return v[:, :, a:b, :]
        
        should_be_tokens = mx.array([1, 2, 4, 7, 8, 9])
        should_be_kv = mx.concatenate(
            [
                idx(cache_kv, 0, 2),
                idx(cache_kv, 3, 4),
                idx(cache_kv, 6, 9),
            ],
            axis=2,
        )

        total_reused = cache._truncate_cache(
            prompt_tokens=prompt_tokens,
            common_prefix_len=prefix_len,
            non_prefix_reuse_min_seq_len=1,
        )

        self.assertEqual(total_reused, 4)
        self.assertArrEqual(cache.tokens, should_be_tokens)
        self.assertArrEqual(cache.cache[0].keys, should_be_kv)

        # ensure updating works as intended
        new_kv = self.make_random_kv(1)
        keys, _ = cache.cache[0].update_and_fetch(new_kv, new_kv)
        should_be_kv = mx.concatenate([should_be_kv, new_kv], axis=2)
        self.assertArrEqual(keys, should_be_kv)

        # ensure batch concat works as intended
        new_kv = self.make_random_kv(2)
        keys, _ = cache.cache[0].update_and_fetch(new_kv, new_kv)
        should_be_kv = mx.concatenate([should_be_kv, new_kv], axis=2)
        self.assertArrEqual(keys, should_be_kv)

    def test_update_cache_heavy(self):
        """Test that the cache updates correctly during generation"""
        # TODO(christian-lms): you need to pipe in nonprefix reuse min seq len
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=10)

        # set up pretend cache
        prompt_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        non_prefill_tokens = model_kit.cache_wrapper.update_cache(prompt_tokens, prompt_progress_callback=None, keep=2)
        layer_0_cache = model_kit.cache_wrapper.cache[0]
        from copy import deepcopy
        original_keys = deepcopy(layer_0_cache.state[0])

        # generate a few tokens
        for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=prompt_tokens,
                seed=0,
                max_tokens=2,
                temp=0.0,
                prompt_progress_callback=None,
                keep=2,
            ):
            print(model_kit.cache_wrapper.tokens.tolist())
            print(result.tokens)

        result_tokens = mx.array([1, 2, 6, 7, 8, 9, 10, 4999, 1725, 1725])
        self.assertArrEqual(model_kit.cache_wrapper.tokens, result_tokens)

        _compA = model_kit.cache_wrapper.cache[0]._temporal_order(model_kit.cache_wrapper.cache[0].state[0])
        compA = _compA[..., :7, :]
        print(_compA[0,0,:,:1].tolist())
        compB = mx.concat(
            [original_keys[..., :2, :], original_keys[..., 4:, :]], axis=2)
        self.assertArrEqual(compA, compB)
        print("---  ---")
        
        new_prompt_tokens = mx.array([1, 2, 8, 9, 10, 4999, 1725, 1725])
        for result in create_generator(
                model_kit=model_kit,
                prompt_tokens=new_prompt_tokens,
                seed=0,
                max_tokens=2,
                temp=0.0,
                prompt_progress_callback=None,
                keep=2,
            ):
            self.assertEqual(len(model_kit.cache_wrapper.tokens), model_kit.cache_wrapper.cache[0].state[0].shape[2])
            print(f"HOASDOSIADN {result.tokens}")
            print(model_kit.cache_wrapper.tokens.tolist())
            print(result.tokens)
        
        print(model_kit.cache_wrapper.tokens.tolist())
        new_result_tokens = mx.array([1, 2, 9, 10, 4999, 1725, 1725, 21002, 1177, 1177])
        self.assertArrEqual(model_kit.cache_wrapper.tokens, new_result_tokens)
        
        _compC = model_kit.cache_wrapper.cache[0]._temporal_order(model_kit.cache_wrapper.cache[0].state[0])
        compC = _compC[..., :3, :]
        print(_compC[0,0,:,:1].tolist())
        print(original_keys[0,0,:,:1].tolist())
        compD = mx.concat(
            [original_keys[..., :2, :], original_keys[..., 8:, :]], axis=2)
        self.assertArrEqual(compC, compD)
        compE = _compC[..., 3:6, :]
        compF = _compA[..., 7:, :]
        print("---  ---")
        print(_compC[0,0,2:5,:1].tolist())
        print(_compA[0,0,7:,:1].tolist())
        self.assertArrEqual(compE, compF)
        raise ValueError()


if __name__ == "__main__":
    unittest.main(verbosity=2)
