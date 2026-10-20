"""Consistency checks for the precalculated notebook list."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIST_FILE = REPO_ROOT / "docs" / "precalculated_notebooks.txt"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def _precalculated_notebooks() -> list[str]:
    """Notebook names listed in ``docs/precalculated_notebooks.txt``.

    Returns:
        Notebook names without the ``.ipynb`` extension.
    """
    return [
        line.strip()
        for line in LIST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_list_has_no_duplicates() -> None:
    """Each notebook is listed at most once."""
    names = _precalculated_notebooks()
    assert names, f"{LIST_FILE} lists no notebooks"
    assert len(names) == len(set(names)), f"{LIST_FILE} has duplicate entries"


@pytest.mark.parametrize("name", _precalculated_notebooks())
def test_listed_notebook_and_source_exist(name: str) -> None:
    """A listed notebook has both its .ipynb and its jupytext source."""
    assert (NOTEBOOKS_DIR / f"{name}.ipynb").is_file(), (
        f"{name} is listed in {LIST_FILE} but notebooks/{name}.ipynb does not exist"
    )
    sources = [(NOTEBOOKS_DIR / "src" / f"{name}.{ext}") for ext in ("py", "m")]
    assert any(source.is_file() for source in sources), (
        f"{name} is listed in {LIST_FILE} but has no source in notebooks/src/"
    )


@pytest.mark.parametrize("name", _precalculated_notebooks())
def test_listed_notebook_is_documented(name: str) -> None:
    """A listed notebook appears in the notebooks documentation page."""
    page = (REPO_ROOT / "docs" / "notebooks.rst").read_text(encoding="utf-8")
    assert f"notebooks/{name}" in page, (
        f"{name} is listed in {LIST_FILE} but missing from docs/notebooks.rst"
    )
