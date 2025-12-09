import unittest

from mlx_engine.utils.hardware import PerformanceProfile
from mlx_engine.utils.prompt_processing import plan_prefill_strategy


class TestPrefillGuardrails(unittest.TestCase):
    def test_reason_records_headroom_fallback(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=1024,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=12000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=512 * 1024**2,  # deliberately tiny
            requested_mode=None,
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertIsNotNone(plan.reason)
        self.assertIn("headroom", plan.reason.lower())

    def test_requested_unbounded_falls_back_when_headroom_low(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=1024,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=8000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=512 * 1024**2,
            requested_mode="unbounded",
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertIsNotNone(plan.chunk_size)
        self.assertIsNotNone(plan.reason)
        self.assertIn("headroom", plan.reason.lower())

    def test_headroom_accounts_for_cache_slots(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=512,
            chunk_size_max=4096,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=200000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=1 * 1024**3,
            requested_mode=None,
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertIsNotNone(plan.chunk_size)
        self.assertGreaterEqual(plan.chunk_size, profile.chunk_size_min)
        self.assertLessEqual(plan.chunk_size, profile.chunk_size_max)
        self.assertIsNotNone(plan.reason)
        self.assertTrue(
            any(word in plan.reason.lower() for word in ["slot", "headroom"])
        )

    def test_speculative_decoding_forces_chunked(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=1024,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=4000,
            profile=profile,
            kv_bytes_per_token=2048,
            available_mem_bytes=16 * 1024**3,
            requested_mode=None,
            speculative_required=True,
        )
        self.assertEqual(plan.mode, "chunked")
        self.assertGreaterEqual(plan.chunk_size, profile.chunk_size_min)
        self.assertIsNotNone(plan.reason)
        self.assertIn("speculative", plan.reason.lower())


if __name__ == "__main__":
    unittest.main()
