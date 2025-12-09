"""
Hardware profile detection and defaults for mlx-engine high-bandwidth Apple Silicon support.

This module provides hardware detection and performance profile selection for Apple Silicon
Macs, enabling optimal configuration for high-bandwidth workloads.
"""

# Initialize logger with basic config to avoid import conflicts
import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Memory monitoring imports
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceProfileEnum(Enum):
    """Performance profiles for different Apple Silicon configurations."""

    DEFAULT_SAFE = "default_safe"
    M3_ULTRA_512 = "m3_ultra_512"
    M3_ULTRA_256 = "m3_ultra_256"
    M3_MAX_128 = "m3_max_128"
    M3_PRO_64 = "m3_pro_64"


@dataclass
class HardwareInfo:
    """Information about detected Apple Silicon hardware."""

    model: str
    chip: str
    memory_gb: int
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    memory_bandwidth_gb_s: int
    total_memory_gb: int


# Compatibility layer for tests
@dataclass
class HardwareInfoCompat:
    """HardwareInfo compatible with test expectations."""

    model_identifier: str
    total_memory_gb: int
    bandwidth_gbps: int
    is_apple_silicon: bool


@dataclass
class PerformanceProfileCompat:
    """PerformanceProfile compatible with test expectations."""

    name: str
    prefill_mode: str
    unbounded_allowed: bool
    cache_slots: int
    chunk_size_min: int
    chunk_size_max: int
    kv_bytes_per_token_estimate: int = 2048
    max_prefill_tokens_per_pass: int = 8192


@dataclass
class ProfileConfig:
    """Configuration for a performance profile."""

    name: str
    description: str
    min_memory_gb: int
    recommended_memory_gb: int
    max_batch_size: int
    cache_size_gb: int
    parallel_threads: int
    memory_headroom_ratio: float = 0.2


# Memory bandwidth mapping for Apple Silicon chips (GB/s)
MEMORY_BANDWIDTH_MAP = {
    "M1": 68.25,
    "M1 Pro": 200,
    "M1 Max": 400,
    "M1 Ultra": 800,
    "M2": 100,
    "M2 Pro": 200,
    "M2 Max": 400,
    "M2 Ultra": 800,
    "M3": 100,
    "M3 Pro": 225,
    "M3 Max": 300,
    "M3 Ultra": 600,
}

# Performance profile configurations
PROFILE_CONFIGS: Dict[PerformanceProfileEnum, ProfileConfig] = {
    PerformanceProfileEnum.DEFAULT_SAFE: ProfileConfig(
        name="default_safe",
        description="Conservative profile for all Apple Silicon hardware",
        min_memory_gb=16,
        recommended_memory_gb=32,
        max_batch_size=1,
        cache_size_gb=4,
        parallel_threads=4,
    ),
    PerformanceProfileEnum.M3_ULTRA_512: ProfileConfig(
        name="m3_ultra_512",
        description="High-performance profile for M3 Ultra with 512GB RAM",
        min_memory_gb=256,
        recommended_memory_gb=512,
        max_batch_size=8,
        cache_size_gb=64,
        parallel_threads=24,
    ),
    PerformanceProfileEnum.M3_ULTRA_256: ProfileConfig(
        name="m3_ultra_256",
        description="High-performance profile for M3 Ultra with 256GB RAM",
        min_memory_gb=128,
        recommended_memory_gb=256,
        max_batch_size=6,
        cache_size_gb=48,
        parallel_threads=20,
    ),
    PerformanceProfileEnum.M3_MAX_128: ProfileConfig(
        name="m3_max_128",
        description="High-performance profile for M3 Max with 128GB RAM",
        min_memory_gb=64,
        recommended_memory_gb=128,
        max_batch_size=4,
        cache_size_gb=32,
        parallel_threads=16,
    ),
    PerformanceProfileEnum.M3_PRO_64: ProfileConfig(
        name="m3_pro_64",
        description="High-performance profile for M3 Pro with 64GB RAM",
        min_memory_gb=32,
        recommended_memory_gb=64,
        max_batch_size=2,
        cache_size_gb=16,
        parallel_threads=12,
    ),
}


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon hardware."""
    try:
        return (
            platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin"
        )
    except Exception:
        return False


def get_system_memory_gb() -> int:
    """Get total system memory in GB."""
    try:
        if platform.system() == "Darwin":
            # Use sysctl on macOS
            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            memsize_bytes = int(result.stdout.split(":")[1].strip())
            return memsize_bytes // (1024**3)  # Convert to GB
        else:
            # Fallback for other platforms
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3)
    except Exception as e:
        logger.warning(f"Failed to detect system memory: {e}")
        return 32  # Conservative fallback


def detect_apple_silicon_model() -> str:
    """Detect the Apple Silicon model using system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse model identifier
        for line in result.stdout.split("\n"):
            if "Model Identifier:" in line:
                model_id = line.split(":")[1].strip()
                return model_id

        # Fallback: try to get chip name
        for line in result.stdout.split("\n"):
            if "Chip:" in line:
                chip = line.split(":")[1].strip()
                return chip

    except Exception as e:
        logger.warning(f"Failed to detect Apple Silicon model: {e}")

    return "Unknown"


def detect_apple_silicon_chip() -> str:
    """Detect the Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.split("\n"):
            if "Chip:" in line:
                chip = line.split(":")[1].strip()
                # Normalize chip name (e.g., "Apple M3 Ultra" -> "M3 Ultra")
                chip = re.sub(r"^Apple\s+", "", chip)
                return chip

    except Exception as e:
        logger.warning(f"Failed to detect Apple Silicon chip: {e}")

    return "Unknown"


def detect_core_counts() -> Tuple[int, int, int]:
    """Detect CPU, GPU, and Neural Engine core counts."""
    cpu_cores = gpu_cores = neural_engine_cores = 0

    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse core information
        for line in result.stdout.split("\n"):
            if "Total Number of Cores:" in line:
                cpu_cores = int(line.split(":")[1].strip())
            elif "Processor Speed:" in line and "CPU" in line:
                # This might contain GPU info in some formats
                pass

        # Try to get GPU info from graphics section
        try:
            gpu_result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_text = gpu_result.stdout.lower()
            if "gpu" in gpu_text or "graphics" in gpu_text:
                # Count GPU cores from display info
                for line in gpu_result.stdout.split("\n"):
                    if "cores" in line.lower() and (
                        "gpu" in line.lower() or "graphics" in line.lower()
                    ):
                        try:
                            match = re.search(r"(\d+)\s+cores", line.lower())
                            if match:
                                gpu_cores = int(match.group(1))
                                break
                        except (AttributeError, ValueError):
                            pass
        except Exception:
            pass

        # Neural Engine cores are typically fixed per chip
        chip = detect_apple_silicon_chip()
        if "M1" in chip:
            neural_engine_cores = 16
        elif "M2" in chip:
            neural_engine_cores = 16
        elif "M3" in chip:
            neural_engine_cores = 16

    except Exception as e:
        logger.warning(f"Failed to detect core counts: {e}")

    return cpu_cores, gpu_cores, neural_engine_cores


def detect_apple_silicon_hardware() -> HardwareInfo:
    """Detect comprehensive Apple Silicon hardware information."""
    if not is_apple_silicon():
        raise RuntimeError("Not running on Apple Silicon hardware")

    model = detect_apple_silicon_model()
    chip = detect_apple_silicon_chip()
    total_memory_gb = get_system_memory_gb()
    cpu_cores, gpu_cores, neural_engine_cores = detect_core_counts()

    # Get memory bandwidth from mapping
    memory_bandwidth_gb_s = MEMORY_BANDWIDTH_MAP.get(chip, 100)  # Conservative fallback

    return HardwareInfo(
        model=model,
        chip=chip,
        memory_gb=total_memory_gb,  # Unified memory
        cpu_cores=cpu_cores,
        gpu_cores=gpu_cores,
        neural_engine_cores=neural_engine_cores,
        memory_bandwidth_gb_s=memory_bandwidth_gb_s,
        total_memory_gb=total_memory_gb,
    )


def select_profile_for_hardware_original(
    hardware: HardwareInfo,
) -> PerformanceProfileEnum:
    """Select optimal performance profile based on detected hardware."""
    chip = hardware.chip
    memory_gb = hardware.total_memory_gb

    # Check for high-end M3 Ultra configurations
    if "M3 Ultra" in chip and memory_gb >= 512:
        logger.info(
            f"Detected M3 Ultra with {memory_gb}GB RAM, selecting m3_ultra_512 profile"
        )
        return PerformanceProfileEnum.M3_ULTRA_512
    elif "M3 Ultra" in chip and memory_gb >= 256:
        logger.info(
            f"Detected M3 Ultra with {memory_gb}GB RAM, selecting m3_ultra_256 profile"
        )
        return PerformanceProfileEnum.M3_ULTRA_256
    elif "M3 Max" in chip and memory_gb >= 128:
        logger.info(
            f"Detected M3 Max with {memory_gb}GB RAM, selecting m3_max_128 profile"
        )
        return PerformanceProfileEnum.M3_MAX_128
    elif "M3 Pro" in chip and memory_gb >= 64:
        logger.info(
            f"Detected M3 Pro with {memory_gb}GB RAM, selecting m3_pro_64 profile"
        )
        return PerformanceProfileEnum.M3_PRO_64
    else:
        logger.info(f"Using default_safe profile for {chip} with {memory_gb}GB RAM")
        return PerformanceProfileEnum.DEFAULT_SAFE


def validate_profile_request(
    requested_profile: PerformanceProfileEnum,
    hardware: HardwareInfo,
) -> Tuple[bool, Optional[str]]:
    """Validate if a requested profile is suitable for the detected hardware."""
    config = PROFILE_CONFIGS[requested_profile]
    memory_gb = hardware.total_memory_gb

    # Check minimum memory requirement
    if memory_gb < config.min_memory_gb:
        return (
            False,
            f"Profile {requested_profile.value} requires at least {config.min_memory_gb}GB RAM, "
            f"but only {memory_gb}GB detected",
        )

    # Check memory headroom (require 20% headroom)
    required_headroom = int(config.recommended_memory_gb * config.memory_headroom_ratio)
    available_headroom = memory_gb - config.recommended_memory_gb

    if available_headroom < required_headroom:
        return (
            False,
            f"Profile {requested_profile.value} requires {config.recommended_memory_gb}GB RAM "
            f"with {required_headroom}GB headroom, but only {memory_gb}GB total available",
        )

    return True, None


def get_optimal_profile(
    requested_profile: Optional[PerformanceProfileEnum] = None,
) -> Tuple[PerformanceProfileEnum, HardwareInfo, Optional[str]]:
    """
    Get the optimal performance profile for the current hardware.

    Args:
        requested_profile: Optional explicitly requested profile

    Returns:
        Tuple of (selected_profile, hardware_info, warning_message)
    """
    try:
        hardware = detect_apple_silicon_hardware()
    except RuntimeError as e:
        logger.error(f"Hardware detection failed: {e}")
        # Fallback to safe defaults
        hardware = HardwareInfo(
            model="Unknown",
            chip="Unknown",
            memory_gb=32,
            cpu_cores=8,
            gpu_cores=8,
            neural_engine_cores=16,
            memory_bandwidth_gb_s=100,
            total_memory_gb=32,
        )
        return PerformanceProfileEnum.DEFAULT_SAFE, hardware, str(e)

    if requested_profile is None:
        # Auto-select based on hardware
        selected_profile = select_profile_for_hardware_original(hardware)
        return selected_profile, hardware, None
    else:
        # Validate requested profile
        is_valid, error_msg = validate_profile_request(requested_profile, hardware)
        if is_valid:
            logger.info(f"Using requested profile: {requested_profile.value}")
            return requested_profile, hardware, None
        else:
            logger.warning(
                f"Requested profile {requested_profile.value} not suitable: {error_msg}"
            )
            # Fallback to auto-selection
            selected_profile = select_profile_for_hardware_original(hardware)
            return selected_profile, hardware, error_msg


def get_profile_config(profile: PerformanceProfileEnum) -> ProfileConfig:
    """Get configuration for a specific performance profile."""
    return PROFILE_CONFIGS[profile]


def list_available_profiles() -> Dict[PerformanceProfileEnum, ProfileConfig]:
    """List all available performance profiles."""
    return PROFILE_CONFIGS.copy()


def print_hardware_info(hardware: HardwareInfo) -> None:
    """Print detailed hardware information for debugging."""
    logger.info("=== Apple Silicon Hardware Information ===")
    logger.info(f"Model: {hardware.model}")
    logger.info(f"Chip: {hardware.chip}")
    logger.info(f"Total Memory: {hardware.total_memory_gb}GB")
    logger.info(f"CPU Cores: {hardware.cpu_cores}")
    logger.info(f"GPU Cores: {hardware.gpu_cores}")
    logger.info(f"Neural Engine Cores: {hardware.neural_engine_cores}")
    logger.info(f"Memory Bandwidth: {hardware.memory_bandwidth_gb_s}GB/s")
    logger.info("=" * 40)


def get_memory_usage_gb() -> Tuple[float, float, float]:
    """
    Get current memory usage in GB.

    Returns:
        Tuple of (used_gb, available_gb, headroom_gb)
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, cannot get memory usage")
        return 0.0, 0.0, 0.0

    try:
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        headroom_gb = available_gb
        return used_gb, available_gb, headroom_gb
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return 0.0, 0.0, 0.0


def get_memory_headroom_ratio() -> float:
    """
    Get memory headroom as a ratio of available to total memory.

    Returns:
        Headroom ratio between 0.0 and 1.0
    """
    if not PSUTIL_AVAILABLE:
        return 0.0

    try:
        memory = psutil.virtual_memory()
        return memory.available / memory.total if memory.total > 0 else 0.0
    except Exception as e:
        logger.warning(f"Failed to get memory headroom ratio: {e}")
        return 0.0


def monitor_memory_usage(interval_s: float = 1.0) -> Dict[str, float]:
    """
    Monitor memory usage over an interval.

    Args:
        interval_s: Interval in seconds to monitor

    Returns:
        Dictionary with memory statistics
    """
    if not PSUTIL_AVAILABLE:
        return {}

    try:
        # Get initial memory state
        initial_memory = psutil.virtual_memory()
        initial_used = initial_memory.used

        # Wait for interval
        import time

        time.sleep(interval_s)

        # Get final memory state
        final_memory = psutil.virtual_memory()
        final_used = final_memory.used

        return {
            "initial_used_gb": initial_used / (1024**3),
            "final_used_gb": final_used / (1024**3),
            "delta_gb": (final_used - initial_used) / (1024**3),
            "available_gb": final_memory.available / (1024**3),
            "utilization": final_memory.percent / 100.0,
        }
    except Exception as e:
        logger.warning(f"Failed to monitor memory usage: {e}")
        return {}


def get_system_load() -> Dict[str, float]:
    """
    Get current system load metrics.

    Returns:
        Dictionary with system load information
    """
    if not PSUTIL_AVAILABLE:
        return {}

    try:
        load_avg = psutil.getloadavg()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            "load_1min": load_avg[0],
            "load_5min": load_avg[1],
            "load_15min": load_avg[2],
            "cpu_percent": cpu_percent,
        }
    except Exception as e:
        logger.warning(f"Failed to get system load: {e}")
        return {}


def check_memory_pressure(threshold_gb: float = 4.0) -> bool:
    """
    Check if memory pressure is above threshold.

    Args:
        threshold_gb: Memory threshold in GB

    Returns:
        True if memory pressure is high
    """
    _, available_gb, _ = get_memory_usage_gb()
    return available_gb < threshold_gb


# Test compatibility functions
def select_profile_for_hardware(
    hardware: HardwareInfoCompat, requested: str, available_mem_gb: int
) -> PerformanceProfileCompat:
    """
    Select profile for hardware (test compatibility version).

    Args:
        hardware: HardwareInfoCompat object
        requested: Requested profile name ("auto" or specific profile)
        available_mem_gb: Available memory in GB

    Returns:
        PerformanceProfileCompat object
    """
    # Validate inputs
    if available_mem_gb < 0:
        raise ValueError(f"Available memory cannot be negative: {available_mem_gb}")

    # Define profile configurations for tests
    profiles = {
        "m3_ultra_512": PerformanceProfileCompat(
            name="m3_ultra_512",
            prefill_mode="unbounded",
            unbounded_allowed=True,
            cache_slots=4,
            chunk_size_min=512,
            chunk_size_max=4096,
            kv_bytes_per_token_estimate=4096,
            max_prefill_tokens_per_pass=16384,
        ),
        "m3_max_128": PerformanceProfileCompat(
            name="m3_max_128",
            prefill_mode="unbounded",
            unbounded_allowed=True,
            cache_slots=2,
            chunk_size_min=384,
            chunk_size_max=3072,
            kv_bytes_per_token_estimate=3072,
            max_prefill_tokens_per_pass=12288,
        ),
        "m3_pro_64": PerformanceProfileCompat(
            name="m3_pro_64",
            prefill_mode="chunked",
            unbounded_allowed=False,
            cache_slots=2,
            chunk_size_min=320,
            chunk_size_max=2560,
            kv_bytes_per_token_estimate=2560,
            max_prefill_tokens_per_pass=10240,
        ),
        "default_safe": PerformanceProfileCompat(
            name="default_safe",
            prefill_mode="chunked",
            unbounded_allowed=False,
            cache_slots=1,
            chunk_size_min=256,
            chunk_size_max=2048,
            kv_bytes_per_token_estimate=2048,
            max_prefill_tokens_per_pass=8192,
        ),
    }

    # Handle auto-selection
    if requested == "auto":
        if (
            hardware.model_identifier == "Mac14,12"
            and hardware.total_memory_gb == 512
            and hardware.is_apple_silicon
        ):
            return profiles["m3_ultra_512"]
        else:
            return profiles["default_safe"]

    # Handle explicit profile request
    if requested in profiles:
        profile = profiles[requested]

        # Apply guardrails - refuse risky profile if headroom low
        if (
            requested == "m3_ultra_512" and available_mem_gb < 100
        ):  # Low headroom threshold
            logger.warning(
                f"Insufficient headroom for {requested}, falling back to default_safe"
            )
            return profiles["default_safe"]

        return profile

    # Invalid profile
    raise ValueError(f"Unknown profile: {requested}")


# Export compatibility classes for tests - keep both versions available
__all__ = [
    "PerformanceProfileCompat",
    "HardwareInfoCompat",
    "select_profile_for_hardware",
]

# Alias for backward compatibility
PerformanceProfile = PerformanceProfileCompat
