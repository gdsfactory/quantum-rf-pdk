#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Detect jupytext YAML headers that leaked into a notebook as a visible cell.

jupytext only strips the ``---`` fenced YAML header from a percent-format source when the
fence sits on a line of its own. A comment reflow that joins ``% ---`` with ``% jupyter:``
turns the whole header into an ordinary cell, which is then rendered in the built docs.

Exits 0 when the notebook is clean and 1 when a header cell is found, so it can be used
as a plain predicate from shell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Comment prefixes used by the jupytext formats in this repository.
COMMENT_PREFIXES = ("%", "#")

# Keys that only ever appear together inside a jupytext header.
HEADER_MARKERS = ("jupytext:", "text_representation:")


def _uncomment(line: str) -> str:
    """Strip a leading jupytext comment marker from a line.

    Args:
        line: A single line of cell source.

    Returns:
        The line without its comment prefix and surrounding whitespace.
    """
    stripped = line.strip()
    for prefix in COMMENT_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def has_leaked_header(notebook: Path) -> bool:
    """Check whether any cell of ``notebook`` is a jupytext YAML header.

    Args:
        notebook: Path to a ``.ipynb`` file.

    Returns:
        ``True`` if a cell looks like a jupytext header.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
    for cell in cells:
        source = cell.get("source", [])
        text = source if isinstance(source, str) else "".join(source)
        if not text.strip():
            continue
        first = _uncomment(text.splitlines()[0])
        # The reflowed case merges the fence with the first key, hence startswith.
        if not first.startswith("---"):
            continue
        if all(marker in text for marker in HEADER_MARKERS):
            return True
    return False


def main(argv: list[str]) -> int:
    """Return 1 if any notebook passed on the command line leaks its jupytext header.

    Args:
        argv: Notebook paths.

    Returns:
        Process exit code.
    """
    return int(any(has_leaked_header(Path(arg)) for arg in argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
