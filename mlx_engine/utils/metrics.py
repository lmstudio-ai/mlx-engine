"""
Metrics collection system for mlx-engine high-bandwidth Apple Silicon support.

This module provides comprehensive metrics collection for performance monitoring,
including timing, cache statistics, memory usage, and Metal throughput monitoring.
"""

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a generation session."""

    # Timing metrics
    total_time_s: float = 0.0
    prefill_time_s: float = 0.0
    generation_time_s: float = 0.0
    tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0

    # Token metrics
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    cache_size_gb: float = 0.0
    cache_utilization_ratio: float = 0.0

    # Memory metrics
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_headroom_gb: float = 0.0
    peak_memory_gb: float = 0.0

    # Metal metrics (when available)
    metal_memory_used_gb: float = 0.0
    metal_memory_available_gb: float = 0.0
    metal_throughput_gb_s: Optional[float] = None

    # Decision metrics
    prefill_mode: str = ""
    chunk_size: Optional[int] = None
    total_chunks: int = 0
    strategy_reason: str = ""

    # System metrics
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: Optional[float] = None

    # Enhanced throughput metrics
    instantaneous_tokens_per_second: float = 0.0
    average_tokens_per_second: float = 0.0
    peak_tokens_per_second: float = 0.0
    throughput_variance: float = 0.0

    # GPU resource monitoring
    gpu_memory_utilization_percent: Optional[float] = None
    gpu_compute_utilization_percent: Optional[float] = None
    gpu_power_usage_watts: Optional[float] = None
    gpu_temperature_celsius: Optional[float] = None

    # Resource guardrails
    memory_pressure_warning_threshold: float = 0.8
    gpu_utilization_warning_threshold: float = 0.9
    throughput_degradation_threshold: float = 0.5

    # Metadata
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """Collects and manages performance metrics during generation."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize metrics collector."""
        self.session_id = session_id or f"session_{int(time.time())}"
        self.metrics = PerformanceMetrics(session_id=self.session_id)
        self.start_time = time.time()
        self.prefill_start_time: Optional[float] = None
        self.generation_start_time: Optional[float] = None
        self.peak_memory_gb = 0.0

    def start_prefill_timing(self) -> None:
        """Start timing prefill phase."""
        self.prefill_start_time = time.time()

    def end_prefill_timing(self, prompt_tokens: int) -> None:
        """End timing prefill phase and calculate metrics."""
        if self.prefill_start_time is None:
            return

        prefill_duration = time.time() - self.prefill_start_time
        self.metrics.prefill_time_s = prefill_duration
        self.metrics.prompt_tokens = prompt_tokens

        if prefill_duration > 0:
            self.metrics.prefill_tokens_per_second = prompt_tokens / prefill_duration

    def start_generation_timing(self) -> None:
        """Start timing generation phase."""
        self.generation_start_time = time.time()

    def end_generation_timing(self, generated_tokens: int) -> None:
        """End timing generation phase and calculate metrics."""
        if self.generation_start_time is None:
            return

        generation_duration = time.time() - self.generation_start_time
        self.metrics.generation_time_s = generation_duration
        self.metrics.generated_tokens = generated_tokens
        self.metrics.total_tokens = self.metrics.prompt_tokens + generated_tokens

        # Calculate overall tokens per second
        total_time = self.metrics.prefill_time_s + generation_duration
        if total_time > 0:
            self.metrics.tokens_per_second = self.metrics.total_tokens / total_time

    def finalize_timing(self) -> None:
        """Finalize timing metrics."""
        self.metrics.total_time_s = time.time() - self.start_time

    def update_cache_metrics(
        self,
        hits: int,
        misses: int,
        evictions: int,
        size_gb: float,
        utilization_ratio: float,
    ) -> None:
        """Update cache performance metrics."""
        self.metrics.cache_hits = hits
        self.metrics.cache_misses = misses
        self.metrics.cache_evictions = evictions
        self.metrics.cache_size_gb = size_gb
        self.metrics.cache_utilization_ratio = utilization_ratio

    def update_memory_metrics(self) -> None:
        """Update memory usage metrics."""
        if PSUTIL_AVAILABLE:
            import psutil

            try:
                memory = psutil.virtual_memory()
                self.metrics.memory_used_gb = memory.used / (1024**3)
                self.metrics.memory_available_gb = memory.available / (1024**3)
                self.metrics.memory_headroom_gb = memory.available / (1024**3)

                # Track peak memory
                if self.metrics.memory_used_gb > self.peak_memory_gb:
                    self.peak_memory_gb = self.metrics.memory_used_gb
                    self.metrics.peak_memory_gb = self.peak_memory_gb
            except Exception as e:
                logger.debug(f"Failed to update memory metrics: {e}")

    def update_metal_metrics(self) -> None:
        """Update Metal-specific metrics when available."""
        if MLX_AVAILABLE:
            import mlx.core as mx

            try:
                # Get Metal memory information if available
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                    metal_memory = mx.metal.get_active_memory()
                    self.metrics.metal_memory_used_gb = metal_memory / (1024**3)

                # Try to get memory info through MLX
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_cache_memory"):
                    cache_memory = mx.metal.get_cache_memory()
                    self.metrics.metal_memory_available_gb = cache_memory / (1024**3)

            except Exception as e:
                logger.debug(f"Failed to get Metal metrics: {e}")

    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        if PSUTIL_AVAILABLE:
            import psutil

            # CPU usage
            self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)

            # GPU utilization is platform-specific and may not be available
            try:
                # On macOS with Apple Silicon, GPU metrics might be available through powermetrics
                # Implementation deferred until platform-specific metrics are available
                pass
            except Exception:
                pass

    def update_decision_metrics(
        self, mode: str, chunk_size: Optional[int], total_chunks: int, reason: str
    ) -> None:
        """Update decision-making metrics."""
        self.metrics.prefill_mode = mode
        self.metrics.chunk_size = chunk_size
        self.metrics.total_chunks = total_chunks
        self.metrics.strategy_reason = reason

    def update_gpu_metrics(self) -> None:
        """Update GPU utilization and resource metrics with safe defaults."""
        try:
            # Try to get GPU metrics through system tools on macOS
            if platform.system() == "Darwin":
                self._update_macos_gpu_metrics()
            else:
                # Placeholder for other platforms
                self.metrics.gpu_utilization_percent = None
                self.metrics.gpu_memory_utilization_percent = None
                self.metrics.gpu_compute_utilization_percent = None
                self.metrics.gpu_power_usage_watts = None
                self.metrics.gpu_temperature_celsius = None
        except Exception as e:
            logger.debug(f"Failed to update GPU metrics: {e}")
            # Set safe defaults on error
            self.metrics.gpu_utilization_percent = None
            self.metrics.gpu_memory_utilization_percent = None
            self.metrics.gpu_compute_utilization_percent = None
            self.metrics.gpu_power_usage_watts = None
            self.metrics.gpu_temperature_celsius = None

    def _update_macos_gpu_metrics(self) -> None:
        """Update GPU metrics on macOS using powermetrics or other tools."""
        try:
            # Try to use powermetrics for detailed GPU metrics
            # This requires elevated privileges, so we'll use safe fallbacks

            # GPU utilization check for macOS
            # Implementation would parse powermetrics output when available
            # Currently set to None to indicate metrics aren't available
            self.metrics.gpu_utilization_percent = None
            self.metrics.gpu_memory_utilization_percent = None
            self.metrics.gpu_compute_utilization_percent = None
            self.metrics.gpu_power_usage_watts = None
            self.metrics.gpu_temperature_celsius = None

        except Exception as e:
            logger.debug(f"macOS GPU metrics update failed: {e}")
            # Set safe defaults
            self.metrics.gpu_utilization_percent = None
            self.metrics.gpu_memory_utilization_percent = None
            self.metrics.gpu_compute_utilization_percent = None
            self.metrics.gpu_power_usage_watts = None
            self.metrics.gpu_temperature_celsius = None

    def update_throughput_metrics(
        self, tokens_generated: int, time_window_s: float
    ) -> None:
        """Update throughput metrics with variance tracking."""
        if time_window_s <= 0:
            return

        # Calculate instantaneous throughput
        instantaneous_tps = tokens_generated / time_window_s
        self.metrics.instantaneous_tokens_per_second = instantaneous_tps

        # Update peak throughput
        if instantaneous_tps > self.metrics.peak_tokens_per_second:
            self.metrics.peak_tokens_per_second = instantaneous_tps

        # Calculate running average (simple exponential moving average)
        if self.metrics.average_tokens_per_second == 0.0:
            self.metrics.average_tokens_per_second = instantaneous_tps
        else:
            # Use EMA with alpha = 0.1 for smoothing
            alpha = 0.1
            self.metrics.average_tokens_per_second = (
                alpha * instantaneous_tps
                + (1 - alpha) * self.metrics.average_tokens_per_second
            )

        # Calculate variance (simplified implementation)
        # In a production system, you'd want to maintain a proper sliding window
        diff = abs(instantaneous_tps - self.metrics.average_tokens_per_second)
        self.metrics.throughput_variance = diff

    def check_resource_guardrails(self) -> Dict[str, bool]:
        """Check resource guardrails and return warnings."""
        warnings = {}

        # Memory pressure check
        if PSUTIL_AVAILABLE:
            import psutil

            try:
                memory = psutil.virtual_memory()
                memory_utilization = memory.percent / 100.0
                if memory_utilization > self.metrics.memory_pressure_warning_threshold:
                    warnings["memory_pressure"] = True
                    logger.warning(
                        f"Memory pressure warning: {memory_utilization:.1%} utilization "
                        f"exceeds threshold of {self.metrics.memory_pressure_warning_threshold:.1%}"
                    )
            except Exception as e:
                logger.debug(f"Failed to check memory pressure: {e}")

        # GPU utilization check
        if (
            self.metrics.gpu_utilization_percent is not None
            and self.metrics.gpu_utilization_percent
            > self.metrics.gpu_utilization_warning_threshold * 100
        ):
            warnings["gpu_utilization_high"] = True
            logger.warning(
                f"GPU utilization warning: {self.metrics.gpu_utilization_percent:.1f}% "
                f"exceeds threshold of {self.metrics.gpu_utilization_warning_threshold * 100:.1f}%"
            )

        # Throughput degradation check
        if (
            self.metrics.average_tokens_per_second > 0
            and self.metrics.instantaneous_tokens_per_second > 0
        ):
            degradation_ratio = (
                self.metrics.instantaneous_tokens_per_second
                / self.metrics.average_tokens_per_second
            )
            if degradation_ratio < self.metrics.throughput_degradation_threshold:
                warnings["throughput_degradation"] = True
                logger.warning(
                    f"Throughput degradation warning: current {self.metrics.instantaneous_tokens_per_second:.1f} tokens/s "
                    f"is {degradation_ratio:.1%} of average {self.metrics.average_tokens_per_second:.1f} tokens/s"
                )

        return warnings

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics snapshot."""
        # Update dynamic metrics before returning
        self.update_memory_metrics()
        self.update_metal_metrics()
        self.update_system_metrics()
        self.update_gpu_metrics()
        return self.metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = PerformanceMetrics(session_id=self.session_id)
        self.start_time = time.time()
        self.prefill_start_time = None
        self.generation_start_time = None
        self.peak_memory_gb = 0.0


class MetalThroughputMonitor:
    """Monitor Metal throughput when available."""

    def __init__(self):
        """Initialize Metal throughput monitor."""
        self.is_available = self._check_metal_availability()
        self.measurements: List[Dict[str, float]] = []
        self.start_time: Optional[float] = None
        self.tokens_processed: int = 0
        self.last_measurement_time: Optional[float] = None

    def _check_metal_availability(self) -> bool:
        """Check if Metal throughput monitoring is available."""
        if not MLX_AVAILABLE:
            return False

        try:
            import mlx.core as mx

            # Check if we're running on Metal backend
            if hasattr(mx, "default_device"):
                device = mx.default_device()
                return str(device).startswith("metal")
        except Exception:
            pass

        return False

    def start_measurement(self) -> None:
        """Start throughput measurement."""
        if not self.is_available:
            return

        self.start_time = time.time()
        self.tokens_processed = 0
        self.last_measurement_time = self.start_time

    def end_measurement(self) -> Optional[float]:
        """End measurement and return throughput in GB/s."""
        if not self.is_available or self.start_time is None:
            return None

        end_time = time.time()
        duration = end_time - self.start_time

        if duration <= 0:
            return None

        # Calculate throughput for Metal memory bandwidth
        # Implementation would measure actual memory bandwidth when available
        throughput_gb_s = None

        # Record measurement
        measurement = {
            "timestamp": end_time,
            "duration_s": duration,
            "tokens_processed": self.tokens_processed,
            "throughput_gb_s": throughput_gb_s,
        }
        self.measurements.append(measurement)

        return throughput_gb_s

    def record_tokens(self, token_count: int) -> None:
        """Record token processing for throughput calculation."""
        if self.is_available:
            self.tokens_processed += token_count

    def get_token_throughput(self) -> Optional[float]:
        """Calculate current token throughput in tokens/second."""
        if not self.is_available or self.start_time is None:
            return None

        current_time = time.time()
        duration = current_time - self.start_time

        if duration <= 0:
            return None

        return self.tokens_processed / duration

    def reset_measurements(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.start_time = None
        self.tokens_processed = 0
        self.last_measurement_time = None

    def get_average_throughput(self) -> Optional[float]:
        """Get average throughput from all measurements."""
        if not self.measurements:
            return None

        throughputs = [
            m.get("throughput_gb_s", 0)
            for m in self.measurements
            if "throughput_gb_s" in m
        ]
        return sum(throughputs) / len(throughputs) if throughputs else None


class ResourceMonitor:
    """Comprehensive resource monitoring with guardrails and safe defaults."""

    def __init__(self, check_interval_s: float = 1.0):
        """Initialize resource monitor."""
        self.check_interval_s = check_interval_s
        self.last_check_time = time.time()
        self.warning_history: Dict[str, float] = {}
        self.critical_thresholds = {
            "memory_utilization": 0.95,
            "gpu_utilization": 0.95,
            "temperature_celsius": 85.0,
        }
        self.warning_thresholds = {
            "memory_utilization": 0.8,
            "gpu_utilization": 0.9,
            "temperature_celsius": 75.0,
        }

    def check_resources(self) -> Dict[str, Any]:
        """Perform comprehensive resource check with guardrails."""
        current_time = time.time()

        # Rate limit checks to avoid excessive system calls
        if current_time - self.last_check_time < self.check_interval_s:
            return {}

        self.last_check_time = current_time
        resource_status = {
            "timestamp": current_time,
            "warnings": [],
            "critical": [],
            "metrics": {},
        }

        # Memory check
        if PSUTIL_AVAILABLE:
            try:
                import psutil

                memory = psutil.virtual_memory()
                memory_util = memory.percent / 100.0
                resource_status["metrics"]["memory_utilization"] = memory_util
                resource_status["metrics"]["memory_available_gb"] = memory.available / (
                    1024**3
                )

                if memory_util >= self.critical_thresholds["memory_utilization"]:
                    resource_status["critical"].append(
                        f"Critical memory utilization: {memory_util:.1%}"
                    )
                elif memory_util >= self.warning_thresholds["memory_utilization"]:
                    resource_status["warnings"].append(
                        f"High memory utilization: {memory_util:.1%}"
                    )
            except Exception as e:
                logger.debug(f"Memory check failed: {e}")

        # GPU check for macOS implementation
        try:
            gpu_metrics = self._get_gpu_metrics()
            if gpu_metrics:
                resource_status["metrics"].update(gpu_metrics)

                # Check GPU utilization
                if "gpu_utilization" in gpu_metrics:
                    gpu_util = gpu_metrics["gpu_utilization"] / 100.0
                    if gpu_util >= self.critical_thresholds["gpu_utilization"]:
                        resource_status["critical"].append(
                            f"Critical GPU utilization: {gpu_util:.1%}"
                        )
                    elif gpu_util >= self.warning_thresholds["gpu_utilization"]:
                        resource_status["warnings"].append(
                            f"High GPU utilization: {gpu_util:.1%}"
                        )

                # Check temperature
                if "temperature_celsius" in gpu_metrics:
                    temp = gpu_metrics["temperature_celsius"]
                    if temp >= self.critical_thresholds["temperature_celsius"]:
                        resource_status["critical"].append(
                            f"Critical GPU temperature: {temp:.1f}°C"
                        )
                    elif temp >= self.warning_thresholds["temperature_celsius"]:
                        resource_status["warnings"].append(
                            f"High GPU temperature: {temp:.1f}°C"
                        )
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")

        # Log warnings and critical issues
        for warning in resource_status["warnings"]:
            logger.warning(f"Resource warning: {warning}")

        for critical in resource_status["critical"]:
            logger.error(f"Resource critical: {critical}")

        return resource_status

    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Get GPU metrics with platform-specific implementation."""
        if platform.system() != "Darwin":
            return None

        try:
            # Placeholder for macOS GPU metrics
            # In a real implementation, this would parse powermetrics output
            # For now, return empty dict to indicate check was performed
            return {}
        except Exception as e:
            logger.debug(f"GPU metrics collection failed: {e}")
            return None

    def is_safe_to_proceed(self) -> bool:
        """Check if it's safe to continue based on resource status."""
        status = self.check_resources()
        return len(status.get("critical", [])) == 0


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_global_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()


def create_session_collector(session_id: str) -> MetricsCollector:
    """Create a new metrics collector for a specific session."""
    return MetricsCollector(session_id=session_id)


def create_resource_monitor(check_interval_s: float = 1.0) -> ResourceMonitor:
    """Create a new resource monitor."""
    return ResourceMonitor(check_interval_s=check_interval_s)


def create_throughput_monitor() -> MetalThroughputMonitor:
    """Create a new Metal throughput monitor."""
    return MetalThroughputMonitor()


# Global resource monitor instance
_global_resource_monitor: Optional[ResourceMonitor] = None


def get_global_resource_monitor() -> ResourceMonitor:
    """Get or create the global resource monitor."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
    return _global_resource_monitor


def reset_global_resource_monitor() -> None:
    """Reset the global resource monitor."""
    global _global_resource_monitor
    if _global_resource_monitor is not None:
        _global_resource_monitor = ResourceMonitor()
