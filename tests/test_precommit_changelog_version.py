"""Tests for the check-changelog-version.py pre-commit hook."""

import importlib.util
import typing as t
from pathlib import Path

import pytest

SCRIPT_REL_PATH = Path(".github") / "check-changelog-version.py"


def _load_check() -> t.Any:
    """Import the standalone changelog check as a module.

    It lives in ``.github/`` rather than an importable package, so it is loaded by path.

    Returns:
        The imported module.
    """
    path = Path(__file__).resolve().parent.parent / SCRIPT_REL_PATH
    spec = importlib.util.spec_from_file_location("check_changelog_version", path)
    assert spec is not None, f"Cannot load {path}"
    assert spec.loader is not None, f"No loader for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load_check()


@pytest.mark.parametrize(
    "heading",
    [
        "## [0.4.0] - 2026-09-18",
        "## [0.4.0]",
        "## 0.4.0",
        "## v0.4.0",
    ],
)
def test_accepts_heading_forms(heading: str) -> None:
    """Both Keep a Changelog and bare headings count as an entry."""
    assert CHECK.has_entry(f"# Changelog\n\n{heading}\n\n### Added\n", "0.4.0")


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## [0.3.8] - 2026-05-21\n",
        "# Changelog\n\n## [0.4.0rc1]\n",  # a prerelease is not the release
        "# Changelog\n\nSee [0.4.0] in the releases page.\n",  # not a heading
        "# Changelog\n\n### [0.4.0]\n",  # wrong heading level
    ],
)
def test_rejects_missing_entry(changelog: str) -> None:
    """A changelog without a level-2 heading for the version fails."""
    assert not CHECK.has_entry(changelog, "0.4.0")


def test_version_is_not_matched_as_a_prefix() -> None:
    """``0.4.0`` must not be satisfied by a ``0.4.0.1`` heading."""
    assert not CHECK.has_entry("# Changelog\n\n## [0.4.0.1]\n", "0.4.0")


def test_repository_changelog_documents_current_version() -> None:
    """The repository's own changelog is up to date, i.e. the hook passes."""
    assert CHECK.main() == 0
