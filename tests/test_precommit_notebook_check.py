"""Tests for the check-notebook-sources.sh pre-commit hook."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Get the repository root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def test_check_notebook_sources_passes_with_valid_notebooks(repo_root: Path) -> None:
    """Test that the check passes when all notebooks have source files."""
    script = repo_root / ".github" / "check-notebook-sources.sh"
    assert script.exists(), f"Script not found at {script}"

    result = subprocess.run(  # noqa: S603
        [str(script)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, f"Check failed: {result.stderr}"
    assert "All notebooks have corresponding source files" in result.stdout


def test_check_notebook_sources_fails_with_orphaned_notebook(repo_root: Path) -> None:
    """Test that the check fails when a notebook is missing its source file."""
    script = repo_root / ".github" / "check-notebook-sources.sh"
    assert script.exists(), f"Script not found at {script}"

    # Create a temporary orphaned notebook
    test_notebook = repo_root / "notebooks" / "test_orphaned_temp.ipynb"

    try:
        # Create minimal valid Jupyter notebook
        test_notebook.write_text(
            '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        )

        result = subprocess.run(  # noqa: S603
            [str(script)],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

        assert result.returncode == 1, "Check should fail with orphaned notebook"
        assert "Found 1 notebook(s) without corresponding source files" in result.stderr
        assert "test_orphaned_temp.ipynb" in result.stderr
        assert "notebooks/src/test_orphaned_temp.py" in result.stderr

    finally:
        # Clean up
        if test_notebook.exists():
            test_notebook.unlink()


def test_all_existing_notebooks_have_sources(repo_root: Path) -> None:
    """Verify that all existing .ipynb files have corresponding .py sources."""
    notebooks_dir = repo_root / "notebooks"
    src_dir = notebooks_dir / "src"

    # Find all .ipynb files in notebooks/ (not in subdirectories)
    ipynb_files = list(notebooks_dir.glob("*.ipynb"))

    for ipynb_file in ipynb_files:
        py_source = src_dir / f"{ipynb_file.stem}.py"
        assert py_source.exists(), (
            f"Notebook {ipynb_file.name} is missing its source file at {py_source}"
        )
