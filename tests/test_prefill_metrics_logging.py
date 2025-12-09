import unittest

from mlx_engine.utils.hardware import PerformanceProfile
from mlx_engine.utils.prompt_processing import plan_prefill_strategy


class TestPrefillMetricsLogging(unittest.TestCase):
    def test_plan_reason_includes_mode_and_headroom(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=1024,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=16000,
            profile=profile,
            kv_bytes_per_token=4096,
            available_mem_bytes=2 * 1024**3,
            requested_mode=None,
        )
        self.assertIsNotNone(plan.reason)
        self.assertTrue(
            any(word in plan.reason.lower() for word in ["mode", "headroom", "chunk"])
        )

    def test_plan_reason_records_unbounded_choice(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=1024,
            chunk_size_max=8192,
        )
        plan = plan_prefill_strategy(
            prompt_tokens=2048,
            profile=profile,
            kv_bytes_per_token=1024,
            available_mem_bytes=64 * 1024**3,
            requested_mode=None,
        )
        self.assertEqual(plan.mode, "unbounded")
        self.assertIsNotNone(plan.reason)
        self.assertIn("unbounded", plan.reason.lower())


if __name__ == "__main__":
    unittest.main()
