"""Generate the plate-capacitor dataset with Palace electrostatic solves.

Run ``python -m qpdk.models.datasets.plate_capacitor --help`` for the sweep CLI.
Lookups need only the bundled Parquet data; regeneration needs Gmsh and Palace.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from qpdk import PDK, __version__, logger
from qpdk.cells import plate_capacitor
from qpdk.models.datasets.generate import sweep, write
from qpdk.models.datasets.metadata import Axis, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.table import Dataset
from qpdk.simulation.palace import ElectrostaticSettings, Palace

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


def main() -> None:
    """Generate a dataset using a local Palace installation or container command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/datasets") / NAME,
        help="Directory for the complete Parquet dataset",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("build/palace/plate-capacitor"),
        help="Retained geometry inputs, results and logs; reuse to resume",
    )
    parser.add_argument(
        "--palace-command",
        default="palace -np 4",
        help="Command prefix, including MPI/container options; config filename is appended",
    )
    for name, values in GRID.items():
        parser.add_argument(
            f"--{name}",
            type=float,
            nargs="+",
            default=values,
            help=f"{name.capitalize()} grid in µm",
        )
    parser.add_argument(
        "--mesh-size", type=float, default=0.55, help="Near-metal mesh size in µm"
    )
    parser.add_argument(
        "--domain-pad",
        type=float,
        default=100.0,
        help="Lateral pad, air height and substrate depth in µm",
    )
    parser.add_argument(
        "--save-fields",
        action="store_true",
        help="Save each terminal's fields for ParaView",
    )
    args = parser.parse_args()
    runner = Palace(
        workdir=args.workdir,
        command=tuple(shlex.split(args.palace_command)),
        settings=ElectrostaticSettings(
            near_mesh=args.mesh_size,
            domain_pad=args.domain_pad,
            save_fields=args.save_fields,
        ),
    )
    generate(
        runner, output=args.output, grid={name: getattr(args, name) for name in GRID}
    )


if __name__ == "__main__":
    main()
