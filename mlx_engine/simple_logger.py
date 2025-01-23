import sys


class SimpleLogger:
    """A minimal logger that prepends a prefix to messages and writes to stderr.

    Args:
        prefix (Optional[str]): The prefix to prepend to all log messages in square brackets.
                              If None, no prefix will be added.

    Example:
        >>> logger = SimpleLogger("MyApp")
        >>> logger.info("Hello, world!")  # Outputs: [MyApp][INFO] Hello, world!
        >>> logger = SimpleLogger(None)
        >>> logger.info("Hello, world!")  # Outputs: [INFO] Hello, world!
    """

    def __init__(self, prefix=None):
        self.prefix = prefix

    def info(self, message):
        prefix_str = f"[{self.prefix}]" if self.prefix else ""
        print(f"{prefix_str}[INFO] {message}", file=sys.stderr, flush=True)

    def warn(self, message):
        prefix_str = f"[{self.prefix}]" if self.prefix else ""
        print(f"{prefix_str}[WARN] {message}", file=sys.stderr, flush=True)
