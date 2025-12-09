import unittest

from mlx_engine.utils.hardware import PerformanceProfile
from mlx_engine.utils.prompt_processing import PrefillPlan, plan_prefill_strategy


class TestPromptProcessingHighBW(unittest.TestCase):
    def test_unbounded_selected_when_allowed_and_headroom_high(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=512,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=8000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=64 * 1024**3,
            requested_mode=None,
        )
        self.assertEqual(plan.mode, "unbounded")
        self.assertIsNone(plan.chunk_size)

    def test_falls_back_to_chunked_when_headroom_low(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=512,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=8000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=1 * 1024**3,
            requested_mode=None,
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertIsNotNone(plan.chunk_size)
        self.assertGreaterEqual(plan.chunk_size, profile.chunk_size_min)
        self.assertLessEqual(plan.chunk_size, profile.chunk_size_max)

    def test_chunk_size_clamped_to_bounds(self):
        profile = PerformanceProfile(
            name="default_safe",
            prefill_mode="chunked",
            unbounded_allowed=False,
            cache_slots=1,
            chunk_size_min=256,
            chunk_size_max=2048,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=50000,
            profile=profile,
            kv_bytes_per_token=2048,
            available_mem_bytes=8 * 1024**3,
            requested_mode="chunked",
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertGreaterEqual(plan.chunk_size, 256)
        self.assertLessEqual(plan.chunk_size, 2048)


if __name__ == "__main__":
    unittest.main()
