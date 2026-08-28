"""
pytest configuration for test_pipeline_integrity.py.

--run-slow controls tests marked @pytest.mark.slow -- these re-execute real
pipeline scripts (the Kalman filter, the full trust suite) as subprocesses
and take real time / rewrite output/ files. Skipped by default so a plain
`pytest` run stays a fast, read-only smoke test.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                      help="also run slow tests that re-execute pipeline scripts")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-executes a real pipeline script (opt-in via --run-slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test -- use --run-slow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
