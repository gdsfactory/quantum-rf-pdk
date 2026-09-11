"""Configuration for pytest."""

from __future__ import annotations

import sys

import gdsfactory as gf
import pytest

from qpdk import PDK


def pytest_collection_modifyitems(
    config: pytest.Config,  # ruff: ignore[unused-function-argument]
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


@pytest.fixture(autouse=True)
def preserve_kcl_cells():
    """Fail a test that deletes pre-existing cells from the shared KCLayout.

    ``gf.kcl`` is process-global; deleting cells from it silently invalidates
    cells other code still references (they are rebuilt through the ``@cell``
    cache-purge path instead). Tests must only ever delete cells they created
    themselves.
    """
    before = set(gf.kcl.each_cell_top_down())
    yield
    after = set(gf.kcl.each_cell_top_down())
    deleted = before - after
    assert not deleted, f"Test deleted pre-existing KCLayout cells: {sorted(deleted)}"
