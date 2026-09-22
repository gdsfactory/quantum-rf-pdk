"""GDSFactory+ CLI smoke test for the QPDK project."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import GFP_REQUIRED_ENV

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_gfp_binary() -> str:
    """Return the provisioned GDSFactory+ v2 binary or skip local runs.

    A plain `which("gfp")` hit is not enough: gsim depends on the PyPI
    gdsfactoryplus, whose 1.8.x CLI is a different tool (no `--cwd`, no
    `stats`). Only the v2 SDK that `just fetch-gfp` provisions is the
    binary under test, so validate it through the v2 module import first.

    Raises:
        AssertionError: Unreachable; the skip/fail helpers always raise.
    """
    from conftest import import_gfp_module  # ruff: ignore[import-outside-top-level]

    if os.environ.get("GFP_BIN"):
        return str(Path(os.environ["GFP_BIN"]).resolve())
    if binary := shutil.which("gfp"):
        try:
            # Probe a v2-only submodule: the PyPI 1.8.x package also installs
            # a gfp binary, but it is a different tool (no --cwd, no stats).
            import_gfp_module("gdsfactoryplus.factory_metadata")
        except Exception:  # ruff: ignore[try-except-pass]
            # Not the v2 SDK: the found binary is not the tool under test.
            pass
        else:
            return str(Path(binary).resolve())
    if os.environ.get(GFP_REQUIRED_ENV) == "1":
        pytest.fail("gfp v2 binary not found; run `just fetch-gfp`")
    pytest.skip("gfp v2 binary not found; run `just fetch-gfp`")
    raise AssertionError("unreachable")


@pytest.mark.gfp
def test_gfp_indexes_qpdk() -> None:
    """Load the project settings and index its PDK factories."""
    # cwd= rather than a --cwd flag: the flag exists only in the extension
    # SDK's CLI, while the PyPI gdsfactoryplus binary that gsim pulls in as a
    # dependency does not support it.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [_find_gfp_binary(), "stats"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Files:" in result.stdout
    assert "\nqpdk " in result.stdout
