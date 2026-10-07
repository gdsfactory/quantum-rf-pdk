"""Generate the plate-capacitor dataset with Palace electrostatic solves.

Run ``just generate-dataset datasets/plate_capacitor.toml`` from a checkout.
Lookups need only the bundled Parquet data; regeneration needs Gmsh and Palace.

Accuracy is set by the mesh, not by the 1e-9 checks applied to the stored
matrices. The two plates are mirror images, so :math:`C_{11} = C_{22}` exactly;
the bundled data differ by up to 0.55%, and refining ``near_mesh`` at the grid
center changed the matrix by 0.6-1.7% per step. Treat entries as accurate to
about 1% and check convergence before relying on finer differences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from qpdk import PDK, __version__, logger
from qpdk.cells import plate_capacitor
from qpdk.models.datasets.generate import sweep, write
from qpdk.models.datasets.metadata import Axis, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.table import Dataset

if TYPE_CHECKING:
    from qpdk.simulation.palace import Palace

NAME = "plate_capacitor_palace"
GRID = {
    "length": [40.0, 80.0, 120.0],
    "width": [5.0, 10.0, 20.0],
    "gap": [4.0, 7.0, 10.0],
}
"""Default grid in µm; 27 geometries, with the CPW cross-section fixed."""


def generate(
    runner: Palace,
    *,
    output: Path = Path("build/datasets") / NAME,
    grid: Mapping[str, Sequence[float]] | None = None,
) -> Dataset:
    """Solve the grid and publish a complete, self-describing dataset.

    Interrupted sweeps resume from the runner's completed geometry directories.
    The output is replaced only after all solves and validation succeed.

    Returns:
        The completed dataset.
    """
    PDK.activate()
    metadata = DatasetMetadata(
        name=NAME,
        description="Palace Maxwell capacitance of QPDK plate-capacitor sheets on silicon with a separate coplanar ground frame.",
        axes=(
            Axis(name="length", unit="um"),
            Axis(name="width", unit="um"),
            Axis(name="gap", unit="um"),
        ),
        variants=("cross_section",),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=("o1", "o2"),
        reference_ground="coplanar_ground_frame",
        provenance={
            "cell": "plate_capacitor",
            "qpdk_version": __version__,
            **runner.provenance,
        },
    )

    def solve(length: float, width: float, gap: float, cross_section: str) -> dict:
        component = plate_capacitor(
            length=length, width=width, gap=gap, cross_section=cross_section
        )
        return {
            "maxwell_capacitance": runner.capacitance(
                component, terminals=metadata.terminals
            )
        }

    chosen = GRID if grid is None else grid
    count = 1
    for values in chosen.values():
        count *= len(values)
    logger.info(f"Generating {NAME}: {count} geometries; completed runs are reused")
    frame = sweep(metadata, solve, chosen, {"cross_section": ["cpw"]})
    dataset = write(output, metadata, frame)
    dataset.grid("maxwell_capacitance", cross_section="cpw")
    logger.info(f"Wrote {dataset!r} to {output}")
    return dataset
