#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check that CHANGELOG.md has an entry for the version in pyproject.toml.

The changelog is easy to forget during a release bump, and nothing else in the
repository notices: ``check-required-files`` only asserts that CHANGELOG.md
exists, not that it is current. This hook fails when ``project.version`` in
pyproject.toml has no matching ``## [x.y.z]`` heading in CHANGELOG.md, so the
omission surfaces at commit time rather than after the tag is pushed.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def read_version(pyproject: Path) -> str:
    """Read ``project.version`` from a pyproject.toml file.

    Args:
        pyproject: Path to the pyproject.toml file.

    Returns:
        The declared version string. A missing static version raises ``KeyError``.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


def has_entry(changelog: str, version: str) -> bool:
    """Check whether the changelog has a level-2 heading for ``version``.

    Both the bracketed Keep a Changelog form (``## [0.4.0] - 2026-09-18``) and a
    bare ``## 0.4.0`` are accepted, with or without a leading ``v``.

    Args:
        changelog: Full text of the changelog.
        version: Version string to look for.

    Returns:
        True when a matching heading is present.
    """
    pattern = rf"^##\s+\[?v?{re.escape(version)}\]?(?:\s|$)"
    return re.search(pattern, changelog, flags=re.MULTILINE) is not None


def main() -> int:
    """Run the check.

    Returns:
        0 when the changelog documents the current version, 1 otherwise.
    """
    try:
        version = read_version(PYPROJECT)
    except KeyError:
        sys.stderr.write(
            "pyproject.toml declares no static [project] version — "
            "cannot determine which changelog entry to require.\n"
        )
        return 1

    if has_entry(CHANGELOG.read_text(encoding="utf-8"), version):
        return 0

    sys.stderr.write(
        f"CHANGELOG.md has no entry for version {version}.\n"
        f"\n"
        f"Add a heading for it, for example:\n"
        f"\n"
        f"    ## [{version}] - YYYY-MM-DD\n"
        f"\n"
        f"    ### Added\n"
        f"\n"
        f"    - ...\n"
        f"\n"
        f"A version bump and its changelog entry belong in the same commit.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
