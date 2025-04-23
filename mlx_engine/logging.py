"""
Simple logging utilities for outputting formatted messages to stderr.

This module provides functions for logging messages at different severity levels
(info, warning, error) to standard error. Each function can optionally include
a prefix to identify the source of the message.

Example usage:
    >>> log_info("Application started", "MyApp")  # [MyApp][INFO] Application started
    >>> log_warn("Configuration missing", "MyApp")  # [MyApp][WARN] Configuration missing
    >>> log_error("Failed to connect", "MyApp")  # [MyApp][ERROR] Failed to connect

    The prefix is optional:
    >>> log_info("Process complete")  # [INFO] Process complete
"""

import sys
from typing import Optional


def _format_message(prefix: Optional[str], level: str, message: str) -> str:
    prefix_str = f"[{prefix}]" if prefix else ""
    return f"{prefix_str}[{level}] {message}"


def log_info(message: str, prefix: Optional[str] = None) -> None:
    """Log an info message to stderr."""
    print(_format_message(prefix, "INFO", message), file=sys.stderr, flush=True)


def log_warn(message: str, prefix: Optional[str] = None) -> None:
    """Log a warning message to stderr."""
    print(_format_message(prefix, "WARN", message), file=sys.stderr, flush=True)


def log_error(message: str, prefix: Optional[str] = None) -> None:
    """Log an error message to stderr."""
    print(_format_message(prefix, "ERROR", message), file=sys.stderr, flush=True)
