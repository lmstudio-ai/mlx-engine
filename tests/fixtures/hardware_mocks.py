"""
Hardware mocking framework for testing high-bandwidth features.

Provides comprehensive hardware constraint simulation for testing edge cases
and performance conditions across different Apple Silicon configurations.
"""

import platform
import subprocess
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging


@dataclass
class MockHardwareSpec:
    """Specification for mock hardware configuration."""

    model_identifier: str
    total_memory_gb: int
    bandwidth_gbps: int
    is_apple_silicon: bool
    available_mem_gb: int
    cpu_cores: int = 8
    gpu_cores: int = 8
    neural_engine_cores: int = 16
    chip: str = "Unknown"


class HardwareMockManager:
    """Manages hardware mocking for testing edge cases and constraints."""

    def __init__(self):
        self.active_patches: List[Any] = []
        self.original_functions: Dict[str, Any] = {}

    def mock_m3_ultra_high_memory(self):
        """Mock M3 Ultra with 512GB RAM for high-bandwidth testing."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="Mac14,12",
                total_memory_gb=512,
                bandwidth_gbps=800,
                is_apple_silicon=True,
                available_mem_gb=480,
                chip="M3 Ultra",
                cpu_cores=24,
                gpu_cores=60,
                neural_engine_cores=16,
            )
        )

    def mock_m3_ultra_standard_memory(self):
        """Mock M3 Ultra with 256GB RAM."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="Mac14,12",
                total_memory_gb=256,
                bandwidth_gbps=800,
                is_apple_silicon=True,
                available_mem_gb=220,
                chip="M3 Ultra",
                cpu_cores=24,
                gpu_cores=60,
                neural_engine_cores=16,
            )
        )

    def mock_m3_max_high_memory(self):
        """Mock M3 Max with 128GB RAM."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="Mac15,8",
                total_memory_gb=128,
                bandwidth_gbps=300,
                is_apple_silicon=True,
                available_mem_gb=100,
                chip="M3 Max",
                cpu_cores=16,
                gpu_cores=40,
                neural_engine_cores=16,
            )
        )

    def mock_memory_constrained_environment(self, available_gb: int = 8):
        """Mock environment with very low memory for testing guardrails."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="MacBookPro18,1",  # M1 Pro
                total_memory_gb=32,
                bandwidth_gbps=200,
                is_apple_silicon=True,
                available_mem_gb=available_gb,
                chip="M1 Pro",
                cpu_cores=10,
                gpu_cores=16,
                neural_engine_cores=16,
            )
        )

    def mock_non_apple_hardware(self):
        """Mock non-Apple Silicon hardware for fallback testing."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="CustomPC",
                total_memory_gb=64,
                bandwidth_gbps=100,
                is_apple_silicon=False,
                available_mem_gb=48,
                chip="Intel i9",
                cpu_cores=16,
                gpu_cores=1,
                neural_engine_cores=0,
            )
        )

    def mock_edge_case_memory(self, total_gb: int, available_gb: int):
        """Mock edge case memory configurations."""
        return self._mock_hardware(
            MockHardwareSpec(
                model_identifier="MacEdgeCase",
                total_memory_gb=total_gb,
                bandwidth_gbps=200,
                is_apple_silicon=True,
                available_mem_gb=available_gb,
                chip="M2 Pro",
                cpu_cores=12,
                gpu_cores=19,
                neural_engine_cores=16,
            )
        )

    def _mock_hardware(self, spec: MockHardwareSpec):
        """Create comprehensive hardware mock with specified parameters."""
        return HardwarePatcher(spec, self)

    def cleanup(self):
        """Clean up all active hardware mocks."""
        for patch_obj in self.active_patches:
            try:
                patch_obj.stop()
            except Exception as e:
                logger.warning(f"Failed to stop patch: {e}")
        self.active_patches.clear()
        self.original_functions.clear()


class HardwarePatcher:
    """Individual hardware patcher for specific mock configuration."""

    def __init__(self, spec: MockHardwareSpec, manager: HardwareMockManager):
        self.spec = spec
        self.manager = manager
        self.patches: List[Any] = []

    def __enter__(self):
        """Apply hardware mocks when entering context."""
        self._apply_patches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hardware mocks when exiting context."""
        self._remove_patches()

    def _apply_patches(self):
        """Apply all necessary patches for the hardware specification."""

        # Mock platform detection
        platform_patch = patch(
            "platform.machine",
            return_value="arm64" if self.spec.is_apple_silicon else "x86_64",
        )
        platform_patch.start()
        self.patches.append(platform_patch)

        system_patch = patch(
            "platform.system",
            return_value="Darwin" if self.spec.is_apple_silicon else "Linux",
        )
        system_patch.start()
        self.patches.append(system_patch)

        # Mock system_profiler output for Apple Silicon
        if self.spec.is_apple_silicon:
            profiler_output = self._generate_system_profiler_output()
            profiler_patch = patch(
                "subprocess.run", side_effect=self._mock_subprocess_run(profiler_output)
            )
            profiler_patch.start()
            self.patches.append(profiler_patch)

        # Mock memory detection
        if self.spec.is_apple_silicon:
            sysctl_patch = patch("subprocess.run", side_effect=self._mock_sysctl_run())
            sysctl_patch.start()
            self.patches.append(sysctl_patch)
        else:
            # Mock os.sysconf for non-Apple systems
            sysconf_patch = patch("os.sysconf", side_effect=self._mock_sysconf())
            sysconf_patch.start()
            self.patches.append(sysconf_patch)

        # Mock psutil if available
        try:
            import psutil

            psutil_patch = patch(
                "psutil.virtual_memory", side_effect=self._mock_psutil_memory()
            )
            psutil_patch.start()
            self.patches.append(psutil_patch)
        except ImportError:
            pass

        # Add to manager's active patches
        self.manager.active_patches.extend(self.patches)

    def _remove_patches(self):
        """Remove all applied patches."""
        for patch_obj in self.patches:
            try:
                patch_obj.stop()
            except Exception as e:
                logger.warning(f"Failed to stop patch: {e}")
        self.patches.clear()

        # Remove from manager's active patches
        for patch_obj in self.patches:
            if patch_obj in self.manager.active_patches:
                self.manager.active_patches.remove(patch_obj)

    def _generate_system_profiler_output(self) -> str:
        """Generate realistic system_profiler output for the mock hardware."""
        return f"""
Hardware Overview:

  Model Name: MacBook Pro
  Model Identifier: {self.spec.model_identifier}
  Model Number: Z15G001L7
  Chip: Apple {self.spec.chip}
  Total Number of Cores: {self.spec.cpu_cores} (8 performance and 4 efficiency)
  Memory: {self.spec.total_memory_gb} GB
  System Firmware Version: 8422.141.2
  OS Loader Version: 8422.141.2
  Serial Number (system): XXXXXXXXXX
  Hardware UUID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Provisioning UDID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
"""

    def _mock_subprocess_run(self, profiler_output: str):
        """Mock subprocess.run for system_profiler calls."""

        def mock_run(command, **kwargs):
            if "system_profiler" in command:
                result = MagicMock()
                result.stdout = profiler_output
                result.returncode = 0
                return result
            elif "sysctl" in command and "hw.memsize" in command:
                result = MagicMock()
                result.stdout = f"hw.memsize: {self.spec.total_memory_gb * 1024**3}"
                result.returncode = 0
                return result
            else:
                # Fallback to original subprocess.run for other calls
                import subprocess

                return subprocess.run(command, **kwargs)

        return mock_run

    def _mock_sysctl_run(self):
        """Mock sysctl calls for memory detection."""

        def mock_run(command, **kwargs):
            if "hw.memsize" in command:
                result = MagicMock()
                result.stdout = f"hw.memsize: {self.spec.total_memory_gb * 1024**3}"
                result.returncode = 0
                return result
            else:
                import subprocess

                return subprocess.run(command, **kwargs)

        return mock_run

    def _mock_sysconf(self):
        """Mock os.sysconf for non-Apple systems."""

        def mock_sysconf(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            elif name == "SC_PHYS_PAGES":
                return (self.spec.total_memory_gb * 1024**3) // 4096
            else:
                import os

                return os.sysconf(name)

        return mock_sysconf

    def _mock_psutil_memory(self):
        """Mock psutil.virtual_memory for memory usage reporting."""

        def mock_virtual_memory():
            mock_memory = MagicMock()
            mock_memory.total = self.spec.total_memory_gb * 1024**3
            mock_memory.available = self.spec.available_mem_gb * 1024**3
            mock_memory.used = (
                self.spec.total_memory_gb - self.spec.available_mem_gb
            ) * 1024**3
            mock_memory.percent = (mock_memory.used / mock_memory.total) * 100
            return mock_memory

        return mock_virtual_memory


class MemoryConstraintSimulator:
    """Simulates various memory constraint scenarios for testing."""

    def __init__(self):
        self.original_functions = {}
        self.active_patches = []

    def simulate_memory_pressure(self, pressure_level: float):
        """
        Simulate memory pressure (0.0 to 1.0).
        0.0 = no pressure, 1.0 = extreme pressure
        """
        if not 0.0 <= pressure_level <= 1.0:
            raise ValueError("pressure_level must be between 0.0 and 1.0")

        # Mock psutil to report low available memory
        def mock_virtual_memory():
            mock_memory = MagicMock()
            total_memory = 64 * 1024**3  # 64GB total
            available_memory = int(total_memory * (1.0 - pressure_level))
            mock_memory.total = total_memory
            mock_memory.available = available_memory
            mock_memory.used = total_memory - available_memory
            mock_memory.percent = pressure_level * 100
            return mock_memory

        try:
            psutil_patch = patch("psutil.virtual_memory", mock_virtual_memory)
            psutil_patch.start()
            self.active_patches.append(psutil_patch)
        except ImportError:
            pass

        return self

    def simulate_memory_fragmentation(self, fragmentation_ratio: float):
        """
        Simulate memory fragmentation affecting large allocations.
        fragmentation_ratio: 0.0 = no fragmentation, 1.0 = severe fragmentation
        """
        if not 0.0 <= fragmentation_ratio <= 1.0:
            raise ValueError("fragmentation_ratio must be between 0.0 and 1.0")

        # Mock mlx memory allocation to simulate fragmentation
        def mock_get_active_memory():
            # Return high memory usage to simulate fragmentation
            base_memory = 32 * 1024**3  # 32GB base
            fragmented_memory = int(base_memory * fragmentation_ratio)
            return base_memory + fragmented_memory

        try:
            mlx_patch = patch(
                "mlx.core.metal.get_active_memory", mock_get_active_memory
            )
            mlx_patch.start()
            self.active_patches.append(mlx_patch)
        except (ImportError, AttributeError):
            pass

        return self

    def simulate_bandwidth_throttling(self, throttle_factor: float):
        """
        Simulate memory bandwidth throttling.
        throttle_factor: 1.0 = full bandwidth, 0.1 = 10% bandwidth
        """
        if not 0.1 <= throttle_factor <= 1.0:
            raise ValueError("throttle_factor must be between 0.1 and 1.0")

        # This would require patching MLX internals for accurate simulation
        # For now, we'll log the throttling for test verification
        logger.info(f"Simulating bandwidth throttling at {throttle_factor * 100:.1f}%")

        return self

    def cleanup(self):
        """Clean up all memory constraint simulations."""
        for patch_obj in self.active_patches:
            try:
                patch_obj.stop()
            except Exception as e:
                logger.warning(f"Failed to stop memory constraint patch: {e}")
        self.active_patches.clear()


# Helper functions for creating fixtures in test files
def create_hardware_mock_fixture():
    """Create a hardware mock fixture for pytest."""

    def fixture():
        manager = HardwareMockManager()
        yield manager
        manager.cleanup()

    return fixture


def create_memory_simulator_fixture():
    """Create a memory simulator fixture for pytest."""

    def fixture():
        simulator = MemoryConstraintSimulator()
        yield simulator
        simulator.cleanup()

    return fixture


# Test presets for common scenarios
class HardwareTestPresets:
    """Predefined hardware test presets for common scenarios."""

    @staticmethod
    def m3_ultra_optimal():
        """Optimal M3 Ultra configuration for high-bandwidth testing."""
        return MockHardwareSpec(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
            available_mem_gb=480,
            chip="M3 Ultra",
            cpu_cores=24,
            gpu_cores=60,
            neural_engine_cores=16,
        )

    @staticmethod
    def m3_ultra_limited_memory():
        """M3 Ultra with limited memory for guardrail testing."""
        return MockHardwareSpec(
            model_identifier="Mac14,12",
            total_memory_gb=512,
            bandwidth_gbps=800,
            is_apple_silicon=True,
            available_mem_gb=64,  # Low available memory
            chip="M3 Ultra",
            cpu_cores=24,
            gpu_cores=60,
            neural_engine_cores=16,
        )

    @staticmethod
    def memory_constrained():
        """Memory-constrained environment for fallback testing."""
        return MockHardwareSpec(
            model_identifier="MacBookPro18,1",
            total_memory_gb=32,
            bandwidth_gbps=200,
            is_apple_silicon=True,
            available_mem_gb=4,  # Very low available memory
            chip="M1 Pro",
            cpu_cores=10,
            gpu_cores=16,
            neural_engine_cores=16,
        )

    @staticmethod
    def non_apple_fallback():
        """Non-Apple hardware for fallback testing."""
        return MockHardwareSpec(
            model_identifier="CustomPC",
            total_memory_gb=64,
            bandwidth_gbps=100,
            is_apple_silicon=False,
            available_mem_gb=48,
            chip="Intel i9",
            cpu_cores=16,
            gpu_cores=1,
            neural_engine_cores=0,
        )
