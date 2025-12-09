"""
High-bandwidth feature test harness.

Runs the high-bandwidth/M3 Ultra test set (default) or the full pytest suite,
then reports progress as a percentage of failing tests to track implementation
status.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import pytest

HIGH_BW_TESTS: List[str] = [
    "tests/test_hardware_profiles.py",
    "tests/test_hardware_profiles_enhanced.py",
    "tests/test_prompt_processing_high_bw.py",
    "tests/test_prefill_guardrails.py",
    "tests/test_prefill_cli_validation.py",
    "tests/test_prefill_metrics_logging.py",
    "tests/test_progress_synthetic_unbounded.py",
    "tests/test_branching_cache_wrapper.py",
    "tests/test_branching_cache_wrapper_enhanced.py",
    "tests/test_cache_metrics_shape.py",
    "tests/test_vision_branch_cache.py",
    "tests/test_generate_unbounded_flow.py",
    "tests/test_generate_stop_unbounded.py",
]


class ProgressCollector:
    def __init__(self) -> None:
        self.total = 0
        self.failed = 0
        self.skipped = 0
        self.passed = 0
        self._recorded_nodes: set[str] = set()

    def _record_outcome(self, report) -> None:
        # Avoid double-counting the same test node when setup/teardown fail
        if getattr(report, "nodeid", None) in self._recorded_nodes:
            return
        self._recorded_nodes.add(report.nodeid)

        self.total += 1
        if report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1
        else:
            self.passed += 1

    def pytest_collectreport(self, report):  # type: ignore[override]
        if report.failed or report.skipped:
            self._record_outcome(report)

    def pytest_runtest_logreport(self, report):  # type: ignore[override]
        if report.when != "call":
            if report.failed or report.skipped:
                self._record_outcome(report)
            return
        self._record_outcome(report)

    @property
    def failure_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.failed / self.total) * 100.0

    def summary_lines(self) -> List[str]:
        return [
            f"total={self.total}",
            f"passed={self.passed}",
            f"failed={self.failed}",
            f"skipped={self.skipped}",
            f"failure_pct={self.failure_pct:.2f}%",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-BW test harness")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pytest suite instead of high-BW subset",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the high-BW test targets and exit"
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Optional explicit test targets; defaults to high-BW subset when not using --all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("High-BW test order:")
        for t in HIGH_BW_TESTS:
            print(f" - {t}")
        sys.exit(0)

    collector = ProgressCollector()

    if args.all:
        targets = args.tests if args.tests else []
    else:
        targets = args.tests if args.tests else HIGH_BW_TESTS

    # Ensure deterministic ordering and quieter output for TDD
    pytest_args = ["-q"] + targets
    exit_code = pytest.main(pytest_args, plugins=[collector])

    print("\n[high-bw harness]" + " " + ", ".join(collector.summary_lines()))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
