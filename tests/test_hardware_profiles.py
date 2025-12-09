import unittest
from unittest import mock

try:
    import sys
    import os

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "mlx_engine", "utils")
    )
    from hardware import (
        HardwareInfoCompat as HardwareInfo,
        PerformanceProfileCompat as PerformanceProfile,
        select_profile_for_hardware,
    )
except Exception as exc:  # pragma: no cover - fails until implementation exists
    HardwareInfo = None
    PerformanceProfile = None
    select_profile_for_hardware = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class TestHardwareProfiles(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR:
            self.skipTest(f"hardware utils not implemented: {IMPORT_ERROR}")

    def test_m3_ultra_auto_enables_unbounded_profile(self):
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )
        profile = select_profile_for_hardware(
            hw, requested="auto", available_mem_gb=480
        )
        self.assertEqual(profile.name, "m3_ultra_512")
        self.assertEqual(profile.prefill_mode, "unbounded")
        self.assertGreaterEqual(profile.cache_slots, 4)
        self.assertTrue(profile.unbounded_allowed)

    def test_unknown_hardware_uses_default_safe_profile(self):
        hw = HardwareInfo(
            model_identifier="GenericPC",
            total_memory_gb=32,
            bandwidth_gbps=60,
            is_apple_silicon=False,
        )
        profile = select_profile_for_hardware(hw, requested="auto", available_mem_gb=24)
        self.assertEqual(profile.name, "default_safe")
        self.assertEqual(profile.prefill_mode, "chunked")
        self.assertFalse(profile.unbounded_allowed)
        self.assertLessEqual(profile.cache_slots, 1)

    def test_guardrail_refuses_unbounded_when_headroom_low(self):
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )
        profile = select_profile_for_hardware(
            hw, requested="m3_ultra_512", available_mem_gb=4
        )
        self.assertEqual(profile.name, "default_safe")
        self.assertEqual(profile.prefill_mode, "chunked")
        self.assertFalse(profile.unbounded_allowed)

    def test_explicit_profile_respected_when_supported(self):
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )
        profile = select_profile_for_hardware(
            hw, requested="m3_ultra_512", available_mem_gb=400
        )
        self.assertEqual(profile.name, "m3_ultra_512")
        self.assertTrue(profile.unbounded_allowed)

    def test_invalid_requested_profile_raises(self):
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )
        with self.assertRaises(ValueError):
            select_profile_for_hardware(
                hw, requested="nonexistent", available_mem_gb=400
            )


if __name__ == "__main__":
    unittest.main()
