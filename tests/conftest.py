"""Configuration for pytest."""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

import gdsfactory as gf
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
    "package absent" skip and any missing transitive dependency into
    failures (used in the gfp CI job, where ``just test-gfp`` provisions
    everything); otherwise a silently missing dependency would green the CI
    with the SAX/LVS coverage gone.

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
        required = os.environ.get(GFP_REQUIRED_ENV) == "1"
        if missing == "gdsfactoryplus":
            hint = "run `just fetch-gfp` and source build/gfp-vsix/env.sh"
            if required:
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
        if required:
            pytest.fail(
                f"gdsfactoryplus is required (GFP_REQUIRED=1) but its "
                f"dependency {missing!r} is not importable — the SDK "
                "provisioning is broken; run `just fetch-gfp` and source "
                "build/gfp-vsix/env.sh, or install the gdsfactoryplus extra"
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


@pytest.fixture(autouse=True)
def preserve_kcl_cells():
    """Fail a test that deletes pre-existing cells from the shared KCLayout.

    ``gf.kcl`` is process-global; deleting cells from it silently invalidates
    cells other code still references (they are rebuilt through the ``@cell``
    cache-purge path instead). Tests must only ever delete cells they created
    themselves.

    Comparison is by cell name, not index: rebuilding a cell under the same
    name is legitimate (kfactory's ``overwrite_existing`` path, used by the
    gdsfactoryplus layout pipeline) and the replacement gets a fresh cell
    index, so index-based comparison would false-positive on it.
    """
    before = {gf.kcl[ci].name for ci in gf.kcl.each_cell_top_down()}
    yield
    after = {gf.kcl[ci].name for ci in gf.kcl.each_cell_top_down()}
    deleted = before - after
    assert not deleted, f"Test deleted pre-existing KCLayout cells: {sorted(deleted)}"
