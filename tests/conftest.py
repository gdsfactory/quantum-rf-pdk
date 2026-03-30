"""Configuration for pytest."""

from __future__ import annotations

import sys

import pytest

from qpdk import PDK


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Skip tests marked with skip_windows on Windows platform or hfss if dependencies missing."""
    skip_windows = pytest.mark.skip(reason="Not supported on Windows")

    import importlib.util

    has_hfss = importlib.util.find_spec("ansys.aedt.core") is not None
    skip_hfss = pytest.mark.skip(reason="hfss extra not installed")

    for item in items:
        if sys.platform == "win32" and "skip_windows" in item.keywords:
            item.add_marker(skip_windows)
        if not has_hfss and "hfss" in item.keywords:
            item.add_marker(skip_hfss)


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate PDK."""
    PDK.activate()
