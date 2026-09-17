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
    """Return the configured GDSFactory+ binary or skip local runs."""
    if binary := os.environ.get("GFP_BIN") or shutil.which("gfp"):
        return str(Path(binary).resolve())
    if os.environ.get(GFP_REQUIRED_ENV) == "1":
        pytest.fail("gfp binary not found; run `just fetch-gfp`")
    pytest.skip("gfp binary not found; run `just fetch-gfp`")
    raise AssertionError("unreachable")


@pytest.mark.gfp
def test_gfp_indexes_qpdk() -> None:
    """Load the project settings and index its PDK factories."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [_find_gfp_binary(), "--cwd", str(PROJECT_ROOT), "stats"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Files:" in result.stdout
    assert "\nqpdk " in result.stdout
