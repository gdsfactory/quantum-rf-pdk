"""Tests for extracting a local gdsfactoryplus development VSIX."""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

_JUSTFILE = Path(__file__).parents[1] / "justfile"


def _write_vsix(path: Path, *extra_members: str) -> None:
    """Write the smallest archive accepted by the ``fetch-gfp`` recipe."""
    executables = (
        ["gfp.exe", "uv.exe", "rg.exe"]
        if platform.system() == "Windows"
        else ["gfp", "uv", "rg"]
    )
    members = {
        **{
            f"extension/bin/{executable}": b"development binary"
            for executable in executables
        },
        "extension/bin/python/gdsfactoryplus/gdsfactoryplus/__init__.py": b"",
        "extension/bin/python/nyancad/nyancad/__init__.py": b"",
        **dict.fromkeys(extra_members, b"must not escape"),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def _fetch(workdir: Path, vsix: Path) -> subprocess.CompletedProcess[str]:
    """Run ``fetch-gfp`` in an isolated working directory."""
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            "just",
            "--justfile",
            str(_JUSTFILE),
            "--working-directory",
            str(workdir),
            "fetch-gfp",
            str(vsix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_fetch_gfp_accepts_local_path_with_spaces(tmp_path: Path) -> None:
    """Pass local paths as data rather than interpolating them into Python."""
    vsix = tmp_path / "local development.vsix"
    _write_vsix(vsix)

    result = _fetch(tmp_path, vsix)

    assert result.returncode == 0, result.stderr
    executable_names = (
        ["gfp.exe", "uv.exe", "rg.exe"]
        if platform.system() == "Windows"
        else ["gfp", "uv", "rg"]
    )
    for executable_name in executable_names:
        executable = tmp_path / "build/gfp-vsix/bin" / executable_name
        assert executable.is_file()
        if platform.system() != "Windows":
            assert os.access(executable, os.X_OK)
    assert (tmp_path / "build/gfp-vsix/bin/python/gdsfactoryplus").is_dir()
    assert (tmp_path / "build/gfp-vsix/bin/python/nyancad").is_dir()


@pytest.mark.parametrize(
    "member",
    [
        "extension/bin/../../escaped",
        "extension/bin/subdirectory/../../../../escaped",
    ],
)
def test_fetch_gfp_rejects_path_traversal(tmp_path: Path, member: str) -> None:
    """Reject archive members whose normalized path leaves ``extension/bin``."""
    vsix = tmp_path / "malicious.vsix"
    _write_vsix(vsix, member)

    result = _fetch(tmp_path, vsix)

    assert result.returncode != 0
    assert "unsafe VSIX member" in result.stderr
    assert not (tmp_path / "build/escaped").exists()
    assert not (tmp_path / "escaped").exists()


def test_fetch_gfp_rejects_symlink_members(tmp_path: Path) -> None:
    """Reject links instead of materializing their targets as executables."""
    vsix = tmp_path / "malicious-symlink.vsix"
    _write_vsix(vsix)
    link = zipfile.ZipInfo("extension/bin/linked")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(vsix, "a") as archive:
        archive.writestr(link, "../../escaped")

    result = _fetch(tmp_path, vsix)

    assert result.returncode != 0
    assert "unsafe VSIX symlink member" in result.stderr
    assert not (tmp_path / "build/gfp-vsix").exists()


def test_fetch_gfp_keeps_previous_install_after_rejected_archive(
    tmp_path: Path,
) -> None:
    """Do not remove a working extraction until its replacement is valid."""
    valid_vsix = tmp_path / "valid.vsix"
    _write_vsix(valid_vsix)
    accepted = _fetch(tmp_path, valid_vsix)
    assert accepted.returncode == 0, accepted.stderr

    install = tmp_path / "build/gfp-vsix"
    gfp_name = "gfp.exe" if platform.system() == "Windows" else "gfp"
    previous_stamp = (install / "VERSION").read_text()
    previous_binary = (install / "bin" / gfp_name).read_bytes()

    invalid_vsix = tmp_path / "invalid.vsix"
    _write_vsix(invalid_vsix, "extension/bin/../../escaped")
    rejected = _fetch(tmp_path, invalid_vsix)

    assert rejected.returncode != 0
    assert (install / "VERSION").read_text() == previous_stamp
    assert (install / "bin" / gfp_name).read_bytes() == previous_binary
    assert not (tmp_path / "build/escaped").exists()
