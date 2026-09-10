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

    gdsfactoryplus >= 2.0 is not on PyPI; it ships bundled in the public
    GDSFactory+ VS Code extension. ``just test-gfp`` provisions it from the
    extension automatically (see ``fetch-gfp`` in ``tests/test.just``).

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
            hint = "run `just fetch-gfp` and source build/gfp-vsix/env.sh"
            if os.environ.get(GFP_REQUIRED_ENV) == "1":
                pytest.fail(
                    f"gdsfactoryplus is required (GFP_REQUIRED=1) but not "
                    f"importable — {hint}"
                )
            pytest.skip(
                f"gdsfactoryplus not installed ({hint})", allow_module_level=True
            )
        if missing.startswith("gdsfactoryplus"):
            pytest.fail(
                f"gdsfactoryplus is installed but {module!r} is unavailable "
                f"({missing!r} missing) — upstream API changed?"
            )
        pytest.skip(
            f"optional dependency {missing!r} not installed", allow_module_level=True
        )
    raise AssertionError("unreachable")


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
