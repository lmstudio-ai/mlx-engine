import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
import sys
from types import ModuleType

# 1. More Robust Mocking: Instead of manual ModuleType, use `patch.dict` for
# `sys.modules` which is cleaner and scoped. We define a helper to set up mocks.
def setup_turboquant_mocks():
    tq_mock = MagicMock(spec=ModuleType("turboquant_mlx"))
    tq_adaptive_mock = MagicMock(spec=ModuleType("turboquant_mlx.adaptive"))
    tq_patch_mock = MagicMock(spec=ModuleType("turboquant_mlx.patch"))
    
    # Mock some expected classes/functions
    class MockTurboQuantKVCache:
        def __init__(self):
            self.state = mx.array([0])
    
    tq_adaptive_mock.make_adaptive_cache = MagicMock(return_value=[MockTurboQuantKVCache() for _ in range(32)])
    tq_patch_mock.apply_patch = MagicMock()
    
    return tq_mock, tq_adaptive_mock, tq_patch_mock

# We still need to mock them before import of CacheWrapper if we want to test its init
tq_mock, tq_adaptive_mock, tq_patch_mock = setup_turboquant_mocks()
sys.modules["turboquant_mlx"] = tq_mock
sys.modules["turboquant_mlx.adaptive"] = tq_adaptive_mock
sys.modules["turboquant_mlx.patch"] = tq_patch_mock

from mlx_engine.cache_wrapper import CacheWrapper, StopPromptProcessing
from mlx_engine.generate import load_model
from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params

class TestTurboQuant(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        tq_adaptive_mock.make_adaptive_cache.reset_mock()
        tq_patch_mock.apply_patch.reset_mock()

    def test_get_kv_cache_quantization_params_turboquant(self):
        # Valid TurboQuant bits
        bits, group, start = get_kv_cache_quantization_params(kv_bits=3, kv_group_size=None, quantized_kv_start=None, turboquant=True)
        self.assertEqual(bits, 3)
        self.assertIsNone(group)
        self.assertIsNone(start)

        # Invalid TurboQuant bits
        with self.assertRaises(ValueError) as cm:
            get_kv_cache_quantization_params(kv_bits=8, kv_group_size=None, quantized_kv_start=None, turboquant=True)
        self.assertIn("Invalid TurboQuant kv_bits value", str(cm.exception))

    def test_cache_wrapper_turboquant_init(self):
        # Mock adaptive cache to return a known list of mock cache objects
        class MockCacheItem:
            def __init__(self):
                self.state = mx.array([0])
        
        mock_cache = [MockCacheItem() for _ in range(10)]
        tq_adaptive_mock.make_adaptive_cache.return_value = mock_cache
        
        model = MagicMock(spec=nn.Module)
        model.layers = [MagicMock()] * 10
        
        # Initialize CacheWrapper with TurboQuant
        wrapper = CacheWrapper(
            model=model,
            max_kv_size=4096,
            kv_bits=3,
            turboquant=True,
            turboquant_fp16_layers=2,
            turboquant_fused=True,
            chunk_size=2048
        )
        
        # Verify TurboQuant specific calls
        tq_adaptive_mock.make_adaptive_cache.assert_called_with(
            num_layers=10,
            bits=3,
            fp16_layers=2,
            fused=True,
            model=model
        )
        tq_patch_mock.apply_patch.assert_called_once()
        self.assertEqual(wrapper.cache, mock_cache)
        self.assertEqual(wrapper.kv_cache_qtn_params["turboquant"], True)

    @patch("mlx_engine.generate.mlx_lm_load")
    @patch("mlx_engine.generate.ModelKit")
    @patch("mlx_engine.generate.json.loads")
    @patch("mlx_engine.generate.Path.read_text")
    @patch("mlx_engine.generate.sanitize_eos_tokens")
    def test_load_model_turboquant_params(self, mock_sanitize, mock_read_text, mock_json_loads, mock_model_kit, mock_mlx_load):
        mock_json_loads.return_value = {"model_type": "llama"}
        mock_read_text.return_value = "{}"
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        
        # Call load_model with TurboQuant params
        load_model(
            "dummy_path",
            turboquant=True,
            kv_bits=3,
            turboquant_fp16_layers=3,
            max_seq_nums=1 # Ensure we don't use BatchedModelKit
        )
        
        # Verify ModelKit was initialized with TurboQuant params
        mock_model_kit.assert_called()
        args, kwargs = mock_model_kit.call_args
        self.assertEqual(kwargs["turboquant"], True)
        self.assertEqual(kwargs["kv_bits"], 3)
        self.assertEqual(kwargs["turboquant_fp16_layers"], 3)

    def test_cache_wrapper_reset_turboquant(self):
        # Test that cache is correctly reset with TurboQuant when StopPromptProcessing is raised
        model = MagicMock(spec=nn.Module)
        model.layers = [MagicMock()] * 8
        
        # Mocking cache objects to have 'state' attribute
        class MockCacheItem:
            def __init__(self):
                self.state = mx.array([0])
        
        mock_cache_objs = [MockCacheItem() for _ in range(8)]
        tq_adaptive_mock.make_adaptive_cache.return_value = mock_cache_objs

        wrapper = CacheWrapper(
            model=model,
            max_kv_size=None,
            kv_bits=4,
            turboquant=True,
            turboquant_fp16_layers=2,
            turboquant_fused=False,
            chunk_size=1024
        )
        
        # Mock _get_num_tokens_in_cache to return None to trigger full cache reset
        with patch.object(wrapper, "_get_num_tokens_in_cache", return_value=None):
            reporter = MagicMock()
            reporter.update.return_value = False # Cancel processing
            
            with self.assertRaises(StopPromptProcessing):
                # Mock maybe_quantize_kv_cache to avoid issues with mock objects
                with patch("mlx_engine.cache_wrapper.maybe_quantize_kv_cache"):
                    wrapper._prefill(model, wrapper.cache, mx.array([1, 2, 3]), reporter, is_draft=False)
            
            # Verify make_adaptive_cache was called again during reset
            tq_adaptive_mock.make_adaptive_cache.assert_called_with(
                model=model,
                num_layers=8,
                bits=4,
                fused=False,
                fp16_layers=2
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
