#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv run --group docs python - "$repo_root" <<'PY'
from pathlib import Path
import sys

import typst

repo_root = Path(sys.argv[1])
figures = repo_root / "notebooks" / "figures"
figures.mkdir(parents=True, exist_ok=True)

for source in sorted((repo_root / "docs" / "figures").glob("*.typ")):
    if source.stem == "style":
        continue
    svg = typst.compile(
        str(source),
        root=str(repo_root),
        font_paths=[str(repo_root / "build" / "docs-fonts")],
        format="svg",
    )
    (figures / f"{source.stem}.svg").write_bytes(svg + b"\n")
PY
