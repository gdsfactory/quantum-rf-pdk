"""Tests for the check-notebook-sources.sh pre-commit hook."""

import importlib.util
import json
import os
import shutil
import subprocess
import typing as t
from pathlib import Path

import pytest

TEMP_NB = "test_orphaned_temp.ipynb"
SCRIPT_REL_PATH = Path(".github") / "check-notebook-sources.sh"
HEADER_CHECK_REL_PATH = Path(".github") / "check_jupytext_header.py"
SUBPROCESS_TIMEOUT = 60  # seconds

# Minimal valid empty Jupyter notebook
MINIMAL_NOTEBOOK = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'

# A notebook whose first cell is the jupytext YAML header, as produced when a
# comment reflow merges the header's `---` fence into the following line.
LEAKED_HEADER_NOTEBOOK = json.dumps({
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "% --- jupyter:\n",
                "%   jupytext:\n",
                "%     text_representation:\n",
                "%       extension: .m\n",
                "%       format_name: percent\n",
                "% ---\n",
            ],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
})


def _load_header_check() -> t.Any:
    """Import the standalone jupytext-header check as a module.

    It lives in ``.github/`` rather than an importable package, so it is loaded by path.

    Returns:
        The imported module.
    """
    path = Path(__file__).resolve().parent.parent / HEADER_CHECK_REL_PATH
    spec = importlib.util.spec_from_file_location("check_jupytext_header", path)
    assert spec is not None, f"Cannot load {path}"
    assert spec.loader is not None, f"No loader for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_env() -> dict[str, str]:
    """Environment with all GIT_* variables stripped.

    Inherited ``GIT_DIR``/``GIT_WORK_TREE`` (e.g. when pytest is invoked from a
    git hook or ``git bisect run``) would make ``git init`` and the script's
    ``git rev-parse --show-toplevel`` resolve to the caller's repository instead
    of the temporary one.

    Returns:
        Environment mapping suitable for subprocess calls.
    """
    return {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }


def _script_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a throwaway git repo with the check script and a known-good notebooks tree.

    The script runs ``git rev-parse --show-toplevel`` and operates on the notebooks/
    tree of *that* repository, so the only way to isolate the test from the working
    tree is to run it inside a temporary git repository. The tree is populated with
    known-good fixtures so the pass case never depends on the developer's
    working-tree state.

    Returns:
        Path to the temporary git repository root.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / SCRIPT_REL_PATH
    assert script.exists(), f"Script not found at {script}"
    if shutil.which("git") is None:
        pytest.skip(
            "git is not available; check-notebook-sources.sh requires a git repository"
        )

    repo = tmp_path_factory.mktemp("notebook_check_repo")
    (repo / SCRIPT_REL_PATH).parent.mkdir()
    shutil.copy2(script, repo / SCRIPT_REL_PATH)
    shutil.copy2(repo_root / HEADER_CHECK_REL_PATH, repo / HEADER_CHECK_REL_PATH)

    # Known-good fixtures: a Python-kernel notebook with its source and a
    # MATLAB-kernel notebook with its source.
    (repo / "notebooks" / "src").mkdir(parents=True)
    (repo / "notebooks" / "python_notebook.ipynb").write_text(MINIMAL_NOTEBOOK)
    (repo / "notebooks" / "src" / "python_notebook.py").write_text("")
    (repo / "notebooks" / "matlab_notebook.ipynb").write_text(MINIMAL_NOTEBOOK)
    (repo / "notebooks" / "src" / "matlab_notebook.m").write_text("")

    try:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            ["git", "-C", str(repo), "init", "-q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=_git_env(),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        pytest.skip(
            "Could not create temporary git repository for the check: "
            f"{exc}; git stderr: {detail}"
        )
    assert (repo / ".git").is_dir(), f"git init did not create {repo / '.git'}"
    return repo


@pytest.fixture
def script_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary git repository holding the check script and known-good notebooks."""
    return _script_repo(tmp_path_factory)


@pytest.mark.skip_windows
def test_check_notebook_sources_logic(script_repo: Path) -> None:
    """Test the notebook source check script logic (both pass and fail cases)."""
    script = script_repo / SCRIPT_REL_PATH

    # 1. Test that the check passes when all notebooks have source files
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(script)],
        capture_output=True,
        text=True,
        cwd=script_repo,
        timeout=SUBPROCESS_TIMEOUT,
        env=_git_env(),
    )

    assert result.returncode == 0, f"Check failed (pass case): {result.stderr}"
    assert "All notebooks have corresponding source files" in result.stdout

    # 2. Test that the check fails when a notebook is missing its source file.
    #    The orphaned notebook is created inside the temporary repository, never
    #    in the real notebooks/ directory.
    test_notebook = script_repo / "notebooks" / TEMP_NB
    test_notebook.write_text(MINIMAL_NOTEBOOK)

    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(script)],
        capture_output=True,
        text=True,
        cwd=script_repo,
        timeout=SUBPROCESS_TIMEOUT,
        env=_git_env(),
    )

    assert result.returncode == 1, "Check should fail with orphaned notebook"
    assert "Found 1 notebook(s) without corresponding source files" in result.stderr
    assert TEMP_NB in result.stderr
    assert f"notebooks/src/{Path(TEMP_NB).stem}.py" in result.stderr


def test_all_existing_notebooks_have_sources() -> None:
    """Verify that all existing .ipynb files have a corresponding jupytext source.

    Source files use ``.py`` for Python-kernel notebooks and ``.m`` for
    MATLAB-kernel notebooks; either is accepted. This is a read-only filesystem
    check against the real notebooks/ directory.
    """
    repo_root = Path(__file__).resolve().parent.parent
    notebooks_dir = repo_root / "notebooks"
    src_dir = notebooks_dir / "src"
    assert notebooks_dir.is_dir(), f"notebooks/ directory not found at {notebooks_dir}"

    # Find all .ipynb files in notebooks/ (not in subdirectories)
    ipynb_files = sorted(notebooks_dir.glob("*.ipynb"))
    assert ipynb_files, f"No .ipynb files found in {notebooks_dir}"

    for ipynb_file in ipynb_files:
        py_source = src_dir / f"{ipynb_file.stem}.py"
        m_source = src_dir / f"{ipynb_file.stem}.m"
        assert py_source.exists() or m_source.exists(), (
            f"Notebook {ipynb_file.name} is missing its source file "
            f"(expected {py_source} or {m_source})"
        )


@pytest.mark.skip_windows
def test_check_notebook_sources_detects_leaked_jupytext_header(
    script_repo: Path,
) -> None:
    """The check script fails when a notebook contains the jupytext header as a cell."""
    script = script_repo / SCRIPT_REL_PATH
    (script_repo / "notebooks" / "matlab_notebook.ipynb").write_text(
        LEAKED_HEADER_NOTEBOOK
    )

    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(script)],
        capture_output=True,
        text=True,
        cwd=script_repo,
        timeout=SUBPROCESS_TIMEOUT,
        env=_git_env(),
    )

    assert result.returncode == 1, "Check should fail on a leaked jupytext header"
    assert "containing the jupytext YAML header as a cell" in result.stderr
    assert "matlab_notebook.ipynb" in result.stderr


def test_leaked_jupytext_header_detection(tmp_path: Path) -> None:
    """The header detector flags a leaked header and accepts a clean notebook."""
    module = _load_header_check()

    leaked = tmp_path / "leaked.ipynb"
    leaked.write_text(LEAKED_HEADER_NOTEBOOK)
    assert module.has_leaked_header(leaked)

    clean = tmp_path / "clean.ipynb"
    clean.write_text(MINIMAL_NOTEBOOK)
    assert not module.has_leaked_header(clean)


def test_existing_notebooks_do_not_leak_jupytext_header() -> None:
    """No notebook in notebooks/ renders its jupytext YAML header as a cell."""
    module = _load_header_check()
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"

    for ipynb_file in sorted(notebooks_dir.glob("*.ipynb")):
        assert not module.has_leaked_header(ipynb_file), (
            f"{ipynb_file.name} contains the jupytext YAML header as a cell; "
            "check that no formatter reflowed the header in notebooks/src/ and "
            "regenerate the notebook"
        )


def test_matlab_sources_have_reflow_safe_headers() -> None:
    """MATLAB jupytext headers survive the ``matlab-reflow-comments`` hook.

    The hook buffers and refills adjacent comment lines whose inner indent is a single
    space, which would fold the header's ``---`` fence into the next line. Lines with an
    inner indent of zero or of two-or-more spaces are passed through untouched, so every
    header line must use one of those.
    """
    sources_dir = Path(__file__).resolve().parent.parent / "notebooks" / "src"

    for matlab_file in sorted(sources_dir.glob("*.m")):
        lines = matlab_file.read_text(encoding="utf-8").splitlines()
        assert lines, f"{matlab_file.name} is empty"
        assert lines[0].startswith("%"), f"{matlab_file.name} has no header"

        for line in lines[1:]:
            uncommented = line.removeprefix("%")
            inner_indent = len(uncommented) - len(uncommented.lstrip())
            assert inner_indent != 1, (
                f"{matlab_file.name}: header line {line!r} has a single-space inner "
                "indent, so matlab-reflow-comments will merge it into the previous line"
            )
            if uncommented.strip() == "---":
                break
        else:
            pytest.fail(f"{matlab_file.name}: unterminated jupytext YAML header")
