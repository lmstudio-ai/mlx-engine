import sys
from typing import Optional, Literal

LogLevel = Literal["INFO", "WARN", "ERROR"]


def _format_message(prefix: Optional[str], level: str, message: str) -> str:
    prefix_str = f"[{prefix}]" if prefix else ""
    return f"{prefix_str}[{level}] {message}"


def _log(level: LogLevel, message: str, prefix: Optional[str] = None) -> None:
    """Generic log function that outputs to stderr.

    Args:
        level: The log level (INFO, WARN, ERROR)
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets
    """
    print(_format_message(prefix, level, message), file=sys.stderr, flush=True)


def log_info(message: str, prefix: Optional[str] = None) -> None:
    """Log an info message to stderr.

    Args:
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets

    Example:
        >>> log_info("Hello, world!", "MyApp")  # Outputs: [MyApp][INFO] Hello, world!
        >>> log_info("Hello, world!")  # Outputs: [INFO] Hello, world!
    """
    _log("INFO", message, prefix)


def log_warn(message: str, prefix: Optional[str] = None) -> None:
    """Log a warning message to stderr.

    Args:
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets

    Example:
        >>> log_warn("Something went wrong!", "MyApp")  # Outputs: [MyApp][WARN] Something went wrong!
        >>> log_warn("Something went wrong!")  # Outputs: [WARN] Something went wrong!
    """
    _log("WARN", message, prefix)


def log_error(message: str, prefix: Optional[str] = None) -> None:
    """Log an error message to stderr.

    Args:
        message: The message to log
        prefix: Optional prefix to prepend to the message in square brackets

    Example:
        >>> log_error("An error occurred!", "MyApp")  # Outputs: [MyApp][ERROR] An error occurred!
        >>> log_error("An error occurred!")  # Outputs: [ERROR] An error occurred!
    """
    _log("ERROR", message, prefix)
