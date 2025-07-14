import pytest


def pytest_addoption(parser):
    """Add command line option for heavy tests."""
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="run heavy tests (e.g., tests that require large models or long execution time)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "heavy: mark test as heavy (requires --heavy option to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip heavy tests unless --heavy option is provided."""
    if config.getoption("--heavy"):
        # --heavy given in cli: do not skip heavy tests
        return

    skip_heavy = pytest.mark.skip(reason="need --heavy option to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
