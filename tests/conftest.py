"""Configuration for pytest."""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

import pytest

from qpdk import PDK

if TYPE_CHECKING:
    from types import ModuleType

#: When set to "1", tests needing gdsfactoryplus fail instead of skipping.
GFP_REQUIRED_ENV = "GFP_REQUIRED"


def import_gfp_module(module: str = "gdsfactoryplus") -> ModuleType:
    """Import a gdsfactoryplus module, skipping only when the package is absent.

    A missing ``gdsfactoryplus.*`` submodule while the package itself imports
    means upstream changed the API these tests guard against — that must fail
    loudly, never silently skip. Set ``GFP_REQUIRED=1`` to also turn the
    "package absent" skip into a failure (used in the gfp CI job).

    Returns:
        The imported module.

    Raises:
        AssertionError: Unreachable; satisfies the type checker since
            ``pytest.skip``/``pytest.fail`` only raise.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "gdsfactoryplus":
            if os.environ.get(GFP_REQUIRED_ENV) == "1":
                pytest.fail(
                    "gdsfactoryplus is required (GFP_REQUIRED=1) but not installed"
                )
            pytest.skip("gdsfactoryplus not installed")
        if missing.startswith("gdsfactoryplus"):
            pytest.fail(
                f"gdsfactoryplus is installed but {module!r} is unavailable "
                f"({missing!r} missing) — upstream API changed?"
            )
        pytest.skip(f"optional dependency {missing!r} not installed")
    raise AssertionError("unreachable")


@pytest.fixture
def gfp_sdk() -> ModuleType:
    """Provide the gdsfactoryplus SDK top-level module (see import_gfp_module)."""
    return import_gfp_module("gdsfactoryplus")


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
