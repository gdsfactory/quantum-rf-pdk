"""Regenerate the synthetic plate-capacitor dataset.

Stands in for Palace electrostatic output until the sweep runner produces real
extractions. The mutual capacitance is the conformal-mapping formula of
:func:`qpdk.models.capacitor.plate_capacitor_capacitance_analytical`; the
capacitance of each pad to ground is an invented perimeter term, so the result
has the structure of a two-terminal Maxwell matrix without being physics.

Run from the repository root::

    uv run --extra models python qpdk/datasets/data/plate_capacitor_synthetic/generate.py
"""

import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

from qpdk.datasets import Dataset, RunStatus
from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import ε_0
from qpdk.models.math import epsilon_eff

HERE = Path(__file__).parent


def synthetic_maxwell(
    length: float, width: float, gap: float, ep_r: float
) -> np.ndarray:
    """Two-pad Maxwell capacitance matrix in F."""
    mutual = float(
        plate_capacitor_capacitance_analytical(
            length=length, width=width, gap=gap, ep_r=ep_r
        )
    )
    ground = float(ε_0 * epsilon_eff(ep_r) * (length + 2 * width) * 1e-6)
    return np.array([[ground + mutual, -mutual], [-mutual, ground + mutual]])


def main() -> None:
    """Write ``results/part-0000.parquet`` from the manifest grid."""
    dataset = Dataset(HERE)
    manifest = dataset.manifest
    ep_r = manifest.stack["substrate"]["relative_permittivity"]
    terminals = manifest.conventions.terminals
    rows = []
    for length, width, gap in product(*(axis.values for axis in manifest.axes)):
        run_id = hashlib.sha256(
            f"{manifest.recipe.name}:{manifest.recipe.version}:{length}:{width}:{gap}".encode()
        ).hexdigest()[:16]
        maxwell = synthetic_maxwell(length, width, gap, ep_r)
        for (i, row), (j, col) in product(enumerate(terminals), repeat=2):
            rows.append({
                "run_id": run_id,
                "status": RunStatus.OK.value,
                "length": length,
                "width": width,
                "gap": gap,
                "cross_section": "cpw",
                "quantity": "maxwell_capacitance",
                "row": row,
                "col": col,
                "value": maxwell[i, j],
                "value_imag": None,
                "unit": "F",
            })
    for part in dataset.result_files:
        part.unlink()
    dataset.append(pl.DataFrame(rows), part="part-0000")


if __name__ == "__main__":
    main()
