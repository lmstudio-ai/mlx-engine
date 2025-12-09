"""
Enhanced logging setup for mlx-engine with structured logging support.

This module configures standard library logging to output to stderr and provides
structured logging capabilities for metrics and performance data.
Individual modules should get their own loggers using logging.getLogger(__name__).
"""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


def setup_logging():
    """Setup basic logging configuration for mlx_engine."""
    # Silence exceptions that happen within the logger
    logging.raiseExceptions = False

    # Configure root logger for mlx_engine
    logger = logging.getLogger("mlx_engine")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # Create handler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)

    # Simple formatter with logger name and level
    formatter = logging.Formatter("[%(module)s][%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class StructuredLogger:
    """Structured logger for metrics and performance data with verbosity gating."""

    def __init__(self, name: str, verbose: bool = False):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.verbose = verbose
        self.session_id = f"session_{int(time.time())}"

    def log_metrics(self, metrics: Dict[str, Any], level: int = logging.INFO) -> None:
        """Log metrics in structured JSON format if verbose is enabled."""
        if not self.verbose:
            return

        # Add metadata
        log_entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "type": "metrics",
            "data": metrics,
        }

        # Log as JSON
        json_str = json.dumps(log_entry, separators=(",", ":"))
        self.logger.log(level, f"METRICS: {json_str}")

    def log_decision(
        self, decision_type: str, details: Dict[str, Any], level: int = logging.INFO
    ) -> None:
        """Log decision details with redaction (no prompt content)."""
        if not self.verbose:
            return

        # Redact any potentially sensitive content
        redacted_details = self._redact_sensitive_data(details)

        log_entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "type": "decision",
            "decision_type": decision_type,
            "data": redacted_details,
        }

        json_str = json.dumps(log_entry, separators=(",", ":"))
        self.logger.log(level, f"DECISION: {json_str}")

    def log_performance(
        self,
        operation: str,
        duration_s: float,
        tokens: Optional[int] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log performance metrics for an operation."""
        if not self.verbose:
            return

        perf_data = {
            "operation": operation,
            "duration_s": duration_s,
            "tokens_per_second": tokens / duration_s
            if tokens and duration_s > 0
            else None,
        }

        if additional_data:
            perf_data.update(additional_data)

        self.log_metrics(perf_data)

    def log_cache_stats(
        self, hits: int, misses: int, evictions: int, size_gb: float, utilization: float
    ) -> None:
        """Log cache statistics."""
        if not self.verbose:
            return

        cache_data = {
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_evictions": evictions,
            "cache_size_gb": size_gb,
            "cache_utilization": utilization,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0.0,
        }

        self.log_metrics(cache_data)

    def log_memory_usage(
        self,
        used_gb: float,
        available_gb: float,
        headroom_gb: float,
        peak_gb: Optional[float] = None,
    ) -> None:
        """Log memory usage statistics."""
        if not self.verbose:
            return

        memory_data = {
            "memory_used_gb": used_gb,
            "memory_available_gb": available_gb,
            "memory_headroom_gb": headroom_gb,
            "memory_utilization": used_gb / (used_gb + available_gb)
            if (used_gb + available_gb) > 0
            else 0.0,
        }

        if peak_gb is not None:
            memory_data["peak_memory_gb"] = peak_gb

        self.log_metrics(memory_data)

    def log_metal_metrics(
        self, memory_used_gb: float, throughput_gb_s: Optional[float] = None
    ) -> None:
        """Log Metal-specific metrics when available."""
        if not self.verbose:
            return

        metal_data = {"metal_memory_used_gb": memory_used_gb}

        if throughput_gb_s is not None:
            metal_data["metal_throughput_gb_s"] = throughput_gb_s

        self.log_metrics(metal_data)

    def _redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from logging (no prompt content)."""
        sensitive_keys = {"prompt", "text", "content", "input", "message", "query"}

        def redact_value(key: str, value: Any) -> Any:
            if isinstance(key, str) and any(
                sensitive in key.lower() for sensitive in sensitive_keys
            ):
                if isinstance(value, str) and len(value) > 0:
                    return f"[REDACTED: {len(value)} chars]"
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    return f"[REDACTED: {len(value)} items]"
                elif isinstance(value, dict) and len(value) > 0:
                    return f"[REDACTED: {len(value)} keys]"
                else:
                    return "[REDACTED]"
            elif isinstance(value, dict):
                return {k: redact_value(k, v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [
                    redact_value(f"{key}[{i}]", item) for i, item in enumerate(value)
                ]
            else:
                return value

        return {k: redact_value(k, v) for k, v in data.items()}

    def set_verbose(self, verbose: bool) -> None:
        """Set verbosity level."""
        self.verbose = verbose

    def set_session_id(self, session_id: str) -> None:
        """Set session ID for tracking."""
        self.session_id = session_id


def get_structured_logger(name: str, verbose: bool = False) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, verbose)


# Global structured logger instance
_global_structured_logger: Optional[StructuredLogger] = None


def get_global_structured_logger() -> StructuredLogger:
    """Get or create the global structured logger."""
    global _global_structured_logger
    if _global_structured_logger is None:
        _global_structured_logger = StructuredLogger("mlx_engine.global")
    return _global_structured_logger


def set_global_verbose(verbose: bool) -> None:
    """Set verbosity for the global structured logger."""
    get_global_structured_logger().set_verbose(verbose)
