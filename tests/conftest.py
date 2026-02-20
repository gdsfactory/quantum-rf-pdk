"""Configuration for pytest."""

from __future__ import annotations

import sys

import pytest

from qpdk import PDK


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Skip tests marked with skip_windows on Windows platform."""
    if sys.platform == "win32":
        skip_windows = pytest.mark.skip(reason="Not supported on Windows")
        for item in items:
            if "skip_windows" in item.keywords:
                item.add_marker(skip_windows)


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate PDK."""
    PDK.activate()
