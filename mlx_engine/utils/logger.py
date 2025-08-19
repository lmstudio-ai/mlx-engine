"""
Basic logging setup for mlx_engine.

This module configures standard library logging to output to stderr.
Individual modules should get their own loggers using logging.getLogger(__name__).
"""

import logging
import sys


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
