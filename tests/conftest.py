"""Shared pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests that download and use real ML models.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="pass --slow to run model-download tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
