#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Check that each sample schematic matches its pic.yml fallback."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[1]
SAMPLE_DIR = ROOT / "qpdk/samples"


@dataclass(slots=True)
class Circuit:
    """Logical circuit shared by the schematic and layout sources."""

    instances: dict[str, dict[str, Any]]
    connections: set[frozenset[str]]
    ports: dict[str, str]


def _gfp_bin() -> Path:
    """Find the converter bundled with gdsfactoryplus."""
    executable = "gfp.exe" if os.name == "nt" else "gfp"
    candidates = (
        os.environ.get("GFP_BIN"),
        shutil.which("gfp"),
        ROOT / "build/gfp-vsix/bin" / executable,
    )
    for candidate in candidates:
        if candidate and (path := Path(candidate)).is_file():
            return path
    raise FileNotFoundError("gfp binary not found; run `just fetch-gfp` or set GFP_BIN")


def _circuit(document: dict[str, Any]) -> Circuit:
    """Reduce a schematic document to its shared circuit data."""
    instances = {}
    for name, instance in document.get("instances", {}).items():
        component = instance["component"].rsplit(".", 1)[-1]
        # Mosaic groups have no layout instance in pic.yml.
        if component != "group":
            instances[name] = {
                "component": component,
                "settings": instance.get("settings") or {},
            }

    connections = {
        frozenset((left, right))
        for left, right in document.get("connections", {}).items()
    }
    for route in document.get("routes", {}).values():
        connections.update(
            frozenset((left, right)) for left, right in route["links"].items()
        )

    return Circuit(
        instances=instances,
        connections=connections,
        ports=document.get("ports") or {},
    )


def _converted_circuit(gfp_bin: Path, gsch_path: Path) -> Circuit:
    """Convert a gsch source to its logical pic.yml circuit."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [gfp_bin, "scm", "convert", gsch_path, "--to", "picyml"],
        check=True,
        capture_output=True,
        shell=False,
        text=True,
    )
    return _circuit(yaml.safe_load(result.stdout))


def _format(circuit: Circuit) -> list[str]:
    """Format a circuit for a stable diff."""
    document = asdict(circuit)
    document["connections"] = sorted(
        sorted(connection) for connection in circuit.connections
    )
    return json.dumps(document, indent=2, sort_keys=True).splitlines(keepends=True)


def main() -> int:
    """Compare every checked-in schematic source pair."""
    gfp_bin = _gfp_bin()
    gsch_paths = {path.stem: path for path in SAMPLE_DIR.glob("*.gsch")}
    yaml_paths = {
        path.name.removesuffix(".pic.yml"): path
        for path in SAMPLE_DIR.glob("*.pic.yml")
    }
    if missing := sorted(gsch_paths.keys() ^ yaml_paths.keys()):
        sys.stderr.write(f"missing .gsch/.pic.yml pair for: {', '.join(missing)}\n")
        return 1

    failed = False
    for stem, gsch_path in sorted(gsch_paths.items()):
        converted = _converted_circuit(gfp_bin, gsch_path)
        checked_in = _circuit(yaml.safe_load(yaml_paths[stem].read_text()))
        if converted != checked_in:
            failed = True
            sys.stderr.writelines(
                unified_diff(
                    _format(converted),
                    _format(checked_in),
                    fromfile=f"{stem}.gsch (converted)",
                    tofile=f"{stem}.pic.yml",
                )
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
