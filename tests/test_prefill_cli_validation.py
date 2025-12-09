import unittest

from mlx_engine.utils.prompt_processing import (
    PerformanceProfile,
    plan_prefill_strategy,
)


class TestPrefillCliValidation(unittest.TestCase):
    def test_invalid_requested_mode_raises(self):
        profile = PerformanceProfile(
            name="m3_ultra_512",
            prefill_mode="auto",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=512,
            chunk_size_max=4096,
        )
        with self.assertRaises(ValueError):
            plan_prefill_strategy(
                prompt_tokens=1024,
                profile=profile,
                kv_bytes_per_token=2048,
                available_mem_bytes=8 * 1024**3,
                requested_mode="invalid-mode",
            )

    def test_unbounded_requested_when_disallowed_raises(self):
        profile = PerformanceProfile(
            name="default_safe",
            prefill_mode="chunked",
            unbounded_allowed=False,
            cache_slots=1,
            chunk_size_min=256,
            chunk_size_max=2048,
        )
        with self.assertRaises(ValueError):
            plan_prefill_strategy(
                prompt_tokens=4096,
                profile=profile,
                kv_bytes_per_token=1024,
                available_mem_bytes=16 * 1024**3,
                requested_mode="unbounded",
            )


if __name__ == "__main__":
    unittest.main()
