import sys
from typing import Optional


def _format_message(prefix: Optional[str], level: str, message: str) -> str:
    prefix_str = f"[{prefix}]" if prefix else ""
    return f"{prefix_str}[{level}] {message}"


def log_info(message: str, prefix: Optional[str] = None) -> None:
    """Log an info message to stderr.

    Args:
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets

    Example:
        >>> log_info("Hello, world!", "MyApp")  # Outputs: [MyApp][INFO] Hello, world!
        >>> log_info("Hello, world!")  # Outputs: [INFO] Hello, world!
    """
    print(_format_message(prefix, "INFO", message), file=sys.stderr, flush=True)


def log_warn(message: str, prefix: Optional[str] = None) -> None:
    """Log a warning message to stderr.

    Args:
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets

    Example:
        >>> log_warn("Something went wrong!", "MyApp")  # Outputs: [MyApp][WARN] Something went wrong!
        >>> log_warn("Something went wrong!")  # Outputs: [WARN] Something went wrong!
    """
    print(_format_message(prefix, "WARN", message), file=sys.stderr, flush=True)
