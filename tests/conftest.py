"""Configuration for pytest."""

from __future__ import annotations

import sys

import pytest

from qpdk import PDK


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom options."""
    parser.addoption(
        "--run-hfss", action="store_true", default=False, help="run hfss tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    config.addinivalue_line("markers", "hfss: mark test as requiring HFSS")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip tests marked with skip_windows on Windows platform, and hfss unless --run-hfss."""
    skip_hfss = pytest.mark.skip(reason="need --run-hfss option to run")
    run_hfss = config.getoption("--run-hfss")
    
    for item in items:
        if "skip_windows" in item.keywords and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="Not supported on Windows"))
        if "hfss" in item.keywords and not run_hfss:
            item.add_marker(skip_hfss)


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate PDK."""
    PDK.activate()
