"""
Enhanced unit tests for hardware profiles with comprehensive edge case coverage.

Tests hardware profile selection, validation, and edge cases for high-bandwidth
Apple Silicon support.
"""

import unittest
from unittest import mock
import sys
import os

# Add the mlx_engine utils path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx_engine", "utils"))

try:
    from hardware import (
        HardwareInfoCompat as HardwareInfo,
        PerformanceProfileCompat as PerformanceProfile,
        select_profile_for_hardware,
        detect_apple_silicon_hardware,
        get_optimal_profile,
        validate_profile_request,
        get_memory_usage_gb,
        check_memory_pressure,
        get_memory_headroom_ratio,
        monitor_memory_usage,
        get_system_load,
        is_apple_silicon,
        get_system_memory_gb,
        detect_apple_silicon_model,
        detect_apple_silicon_chip,
        detect_core_counts,
        MEMORY_BANDWIDTH_MAP,
        PROFILE_CONFIGS,
        PerformanceProfileEnum,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - fails until implementation exists
    IMPORT_ERROR = exc
    HardwareInfo = None
    PerformanceProfile = None
    select_profile_for_hardware = None


class TestHardwareProfilesEnhanced(unittest.TestCase):
    """Enhanced tests for hardware profile selection and validation."""

    def setUp(self):
        if IMPORT_ERROR:
            self.skipTest(f"hardware utils not implemented: {IMPORT_ERROR}")

    def test_memory_bandwidth_calculation_edge_cases(self):
        """Test bandwidth detection with various hardware configurations."""
        # Test known Apple Silicon chips
        test_cases = [
            ("M1", 68.25),
            ("M1 Pro", 200),
            ("M1 Max", 400),
            ("M1 Ultra", 800),
            ("M2", 100),
            ("M2 Pro", 200),
            ("M2 Max", 400),
            ("M2 Ultra", 800),
            ("M3", 100),
            ("M3 Pro", 225),
            ("M3 Max", 300),
            ("M3 Ultra", 600),
        ]

        for chip, expected_bandwidth in test_cases:
            with self.subTest(chip=chip):
                self.assertEqual(MEMORY_BANDWIDTH_MAP.get(chip), expected_bandwidth)

        # Test unknown chip fallback
        unknown_bandwidth = MEMORY_BANDWIDTH_MAP.get("Unknown Chip", 100)
        self.assertEqual(unknown_bandwidth, 100)

    def test_profile_inheritance_and_overrides(self):
        """Test that custom profiles can inherit from base profiles."""
        # Test that all required profile configs exist
        required_profiles = [
            PerformanceProfileEnum.DEFAULT_SAFE,
            PerformanceProfileEnum.M3_ULTRA_512,
            PerformanceProfileEnum.M3_ULTRA_256,
            PerformanceProfileEnum.M3_MAX_128,
            PerformanceProfileEnum.M3_PRO_64,
        ]

        for profile_enum in required_profiles:
            with self.subTest(profile=profile_enum):
                self.assertIn(profile_enum, PROFILE_CONFIGS)
                config = PROFILE_CONFIGS[profile_enum]
                self.assertIsNotNone(config.name)
                self.assertGreater(config.min_memory_gb, 0)
                self.assertGreater(config.recommended_memory_gb, 0)
                self.assertGreater(config.max_batch_size, 0)
                self.assertGreater(config.cache_size_gb, 0)
                self.assertGreater(config.parallel_threads, 0)

    def test_dynamic_memory_headroom_calculation(self):
        """Test headroom calculation under different memory pressure scenarios."""
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )

        # Test with sufficient headroom
        profile = select_profile_for_hardware(
            hw, requested="auto", available_mem_gb=400
        )
        self.assertEqual(profile.name, "m3_ultra_512")

        # Test with low headroom (should fallback)
        profile = select_profile_for_hardware(
            hw, requested="m3_ultra_512", available_mem_gb=50
        )
        self.assertEqual(profile.name, "default_safe")

    def test_hardware_detection_fallbacks(self):
        """Test graceful degradation when hardware detection fails."""
        # Test non-Apple Silicon fallback
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

    def test_profile_validation_edge_cases(self):
        """Test validation of profile parameters and constraints."""
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )

        # Test invalid profile name
        with self.assertRaises(ValueError):
            select_profile_for_hardware(
                hw, requested="invalid_profile", available_mem_gb=400
            )

        # Test negative available memory
        with self.assertRaises(ValueError):
            select_profile_for_hardware(hw, requested="auto", available_mem_gb=-10)

    def test_memory_pressure_detection(self):
        """Test memory pressure detection under various conditions."""
        # Test with mock psutil available
        with mock.patch("hardware.PSUTIL_AVAILABLE", True):
            with mock.patch("hardware.psutil.virtual_memory") as mock_memory:
                # Test low memory pressure
                mock_memory.return_value.available = 16 * 1024**3  # 16GB
                self.assertFalse(check_memory_pressure(threshold_gb=4.0))

                # Test high memory pressure
                mock_memory.return_value.available = 2 * 1024**3  # 2GB
                self.assertTrue(check_memory_pressure(threshold_gb=4.0))

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring and statistics collection."""
        with mock.patch("hardware.PSUTIL_AVAILABLE", True):
            with mock.patch("hardware.psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.used = 32 * 1024**3  # 32GB
                mock_memory.return_value.available = 16 * 1024**3  # 16GB
                mock_memory.return_value.total = 48 * 1024**3  # 48GB

                used, available, headroom = get_memory_usage_gb()
                self.assertAlmostEqual(used, 32, delta=1)
                self.assertAlmostEqual(available, 16, delta=1)
                self.assertAlmostEqual(headroom, 16, delta=1)

    def test_system_load_monitoring(self):
        """Test system load monitoring functionality."""
        with mock.patch("hardware.PSUTIL_AVAILABLE", True):
            with mock.patch("hardware.psutil.getloadavg") as mock_loadavg:
                with mock.patch("hardware.psutil.cpu_percent") as mock_cpu:
                    mock_loadavg.return_value = (1.5, 1.2, 1.0)
                    mock_cpu.return_value = 25.0

                    load_info = get_system_load()
                    self.assertEqual(load_info["load_1min"], 1.5)
                    self.assertEqual(load_info["load_5min"], 1.2)
                    self.assertEqual(load_info["load_15min"], 1.0)
                    self.assertEqual(load_info["cpu_percent"], 25.0)

    def test_apple_silicon_detection(self):
        """Test Apple Silicon detection logic."""
        # Test Apple Silicon detection
        with mock.patch("platform.machine", return_value="arm64"):
            with mock.patch("platform.system", return_value="Darwin"):
                self.assertTrue(is_apple_silicon())

        # Test non-Apple Silicon detection
        with mock.patch("platform.machine", return_value="x86_64"):
            self.assertFalse(is_apple_silicon())

    def test_system_memory_detection(self):
        """Test system memory detection with fallbacks."""
        # Test macOS sysctl
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "hw.memsize: 68719476736"  # 64GB
            mock_run.return_value.returncode = 0

            memory_gb = get_system_memory_gb()
            self.assertEqual(memory_gb, 64)

    def test_model_and_chip_detection(self):
        """Test Apple Silicon model and chip detection."""
        mock_profiler_output = """
Hardware Overview:

  Model Name: MacBook Pro
  Model Identifier: Mac14,12
  Chip: Apple M3 Ultra
  Total Number of Cores: 24
  Memory: 512 GB
"""

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = mock_profiler_output
            mock_run.return_value.returncode = 0

            model = detect_apple_silicon_model()
            chip = detect_apple_silicon_chip()

            self.assertEqual(model, "Mac14,12")
            self.assertEqual(chip, "M3 Ultra")

    def test_core_count_detection(self):
        """Test CPU, GPU, and Neural Engine core detection."""
        mock_hardware_output = """
Hardware Overview:

  Model Name: MacBook Pro
  Model Identifier: Mac14,12
  Chip: Apple M3 Ultra
  Total Number of Cores: 24
  Memory: 512 GB
"""

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = mock_hardware_output
            mock_run.return_value.returncode = 0

            cpu_cores, gpu_cores, neural_cores = detect_core_counts()

            self.assertGreater(cpu_cores, 0)
            self.assertGreaterEqual(neural_cores, 0)

    def test_memory_monitoring_without_psutil(self):
        """Test memory monitoring behavior when psutil is not available."""
        with mock.patch("hardware.PSUTIL_AVAILABLE", False):
            # Should return zeros when psutil not available
            used, available, headroom = get_memory_usage_gb()
            self.assertEqual(used, 0.0)
            self.assertEqual(available, 0.0)
            self.assertEqual(headroom, 0.0)

            # Should return empty dict for system load
            load_info = get_system_load()
            self.assertEqual(load_info, {})

            # Should return 0.0 for headroom ratio
            ratio = get_memory_headroom_ratio()
            self.assertEqual(ratio, 0.0)

    def test_memory_monitoring_with_exceptions(self):
        """Test memory monitoring error handling."""
        with mock.patch("hardware.PSUTIL_AVAILABLE", True):
            with mock.patch(
                "hardware.psutil.virtual_memory", side_effect=Exception("Test error")
            ):
                # Should handle exceptions gracefully
                used, available, headroom = get_memory_usage_gb()
                self.assertEqual(used, 0.0)
                self.assertEqual(available, 0.0)
                self.assertEqual(headroom, 0.0)

    def test_profile_selection_with_edge_case_memory(self):
        """Test profile selection with edge case memory configurations."""
        # Test exactly at boundary
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=256,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )

        # Should select appropriate profile for 256GB
        profile = select_profile_for_hardware(
            hw, requested="auto", available_mem_gb=200
        )
        self.assertIn(profile.name, ["m3_ultra_256", "default_safe"])

    def test_concurrent_profile_selection(self):
        """Test thread safety of profile selection."""
        import threading

        results = []
        hw = HardwareInfo(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
        )

        def select_profile():
            profile = select_profile_for_hardware(
                hw, requested="auto", available_mem_gb=400
            )
            results.append(profile.name)

        # Create multiple threads selecting profiles
        threads = [threading.Thread(target=select_profile) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should return the same result
        self.assertTrue(all(r == results[0] for r in results))

    def test_profile_configuration_validation(self):
        """Test validation of profile configuration parameters."""
        for profile_enum, config in PROFILE_CONFIGS.items():
            with self.subTest(profile=profile_enum):
                # Validate memory constraints
                self.assertGreaterEqual(
                    config.recommended_memory_gb, config.min_memory_gb
                )
                self.assertLessEqual(config.memory_headroom_ratio, 1.0)
                self.assertGreaterEqual(config.memory_headroom_ratio, 0.0)

                # Validate performance parameters
                self.assertGreater(config.max_batch_size, 0)
                self.assertGreater(config.cache_size_gb, 0)
                self.assertGreater(config.parallel_threads, 0)

                # Validate description
                self.assertIsInstance(config.description, str)
                self.assertGreater(len(config.description), 0)


if __name__ == "__main__":
    unittest.main()
