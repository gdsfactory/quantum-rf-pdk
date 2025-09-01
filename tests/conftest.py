"""Configuration for pytest."""

from __future__ import annotations

import pytest

from qpdk import PDK


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate PDK."""
    PDK.activate()
