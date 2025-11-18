"""Tests for package build configuration."""

import subprocess
import tarfile
from pathlib import Path


def test_sdist_excludes_tests_and_docs():
    """Verify that the source distribution excludes tests and docs directories."""
    # Get the project root
    project_root = Path(__file__).parent.parent

    # Build the package
    subprocess.run(  # noqa: S603, S607
        ["uv", "build"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    # Find the built tarball
    dist_dir = project_root / "dist"
    tarballs = list(dist_dir.glob("*.tar.gz"))
    assert len(tarballs) > 0, "No tarball found in dist/"

    # Get the most recent tarball
    tarball = max(tarballs, key=lambda p: p.stat().st_mtime)

    # Extract the file list from the tarball
    with tarfile.open(tarball, "r:gz") as tar:
        file_list = tar.getnames()

    # Verify tests and docs are not included
    tests_files = [f for f in file_list if "/tests/" in f or f.endswith("/tests")]
    docs_files = [f for f in file_list if "/docs/" in f or f.endswith("/docs")]

    assert len(tests_files) == 0, f"Found tests files in package: {tests_files}"
    assert len(docs_files) == 0, f"Found docs files in package: {docs_files}"
