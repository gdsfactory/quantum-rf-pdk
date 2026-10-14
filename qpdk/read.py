"""Read QPDK layouts from GDSFactory+ schematics."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gdsfactory as gf
import yaml

from qpdk.config import PATH
from qpdk.tech import routing_strategies


def read_gsch(path: str | Path) -> dict[str, Any]:
    """Convert a schematic to a layout netlist using the GDSFactory+ CLI.

    Run ``just fetch-gfp`` in a checkout, put ``gfp`` on PATH, or set
    ``GFP_BIN`` to the converter bundled with the GDSFactory+ extension.
    No generated YAML file is stored alongside the schematic.

    Returns:
        A layout netlist with QPDK component names and editor groups removed.

    Raises:
        FileNotFoundError: If the GDSFactory+ converter cannot be found.
    """
    executable = "gfp.exe" if os.name == "nt" else "gfp"
    candidates = (
        os.environ.get("GFP_BIN"),
        PATH.repo / "build/gfp-vsix/bin" / executable,
        shutil.which("gfp"),
    )
    binary = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if binary is None:
        raise FileNotFoundError(
            "gfp binary not found; run `just fetch-gfp` or set GFP_BIN"
        )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(binary), "scm", "convert", str(path), "--to", "picyml"],
        check=True,
        capture_output=True,
        shell=False,
        text=True,
    )
    document = yaml.safe_load(result.stdout)
    # Editor groups annotate the schematic but have no layout geometry.
    for name, instance in list(document["instances"].items()):
        component = instance["component"].rsplit(".", 1)[-1]
        if component == "group":
            del document["instances"][name]
            document.get("placements", {}).pop(name, None)
        else:
            instance["component"] = component
    return document


def from_gsch(path: str | Path) -> gf.Component:
    """Materialize a schematic with QPDK routing and its source stem as cell name."""
    return gf.read.from_yaml(
        read_gsch(path), name=Path(path).stem, routing_strategies=routing_strategies
    )
