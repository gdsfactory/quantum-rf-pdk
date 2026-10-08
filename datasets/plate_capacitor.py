# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models]",
#   "gsim @ git+https://github.com/gdsfactory/gsim.git@05c6c93cc14522f8a6a78a08b28116e242cdf1c0",
# ]
# [tool.uv.sources]
# qpdk = { path = "..", editable = true }
# ///
"""Generate Maxwell capacitance data with gsim and Palace.

Run ``uv run --script datasets/plate_capacitor.py --help`` from a checkout.
Edit GRID and SETTINGS to define your experiment. The electrodes are perfect
conductor sheets on silicon with a separate coplanar ground frame. Mesh and
domain sensitivity must be checked before relying on small differences.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import gdsfactory as gf
import numpy as np
from shapely import Point, Polygon, box

from datasets._palace import extract, fingerprint, runtime
from qpdk import PDK, logger
from qpdk.cells import plate_capacitor
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write
from qpdk.tech import LAYER, material_properties

NAME = "plate_capacitor_palace"
GRID = {
    "length": [40.0, 80.0, 120.0],
    "width": [5.0, 10.0, 20.0],
    "gap": [4.0, 7.0, 10.0],
}
TERMINALS = ("o1", "o2")


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Lengths in µm; planar perfect conductors at the silicon interface."""

    near_mesh: float = 0.55
    far_mesh: float = 25.0
    domain_pad: float = 100.0
    ground_clearance: float = 10.0
    ground_pad: float = 45.0
    permittivity: float = material_properties["Si"]["relative_permittivity"]
    order: int = 2
    tolerance: float = 1e-9
    save_fields: bool = False

    def __post_init__(self) -> None:
        """Reject an invalid ground frame, mesh or numerical setting."""
        if not 0 < self.ground_clearance < self.ground_pad < self.domain_pad:
            raise ValueError("Require 0 < ground_clearance < ground_pad < domain_pad")
        if not 0 < self.near_mesh <= self.far_mesh or self.permittivity <= 0:
            raise ValueError("Mesh sizes and permittivity must be positive")
        if self.order < 1 or not 0 < self.tolerance < 1:
            raise ValueError("Require order >= 1 and 0 < tolerance < 1")


SETTINGS = Settings()


def _terminal_polygons(
    component: gf.Component, terminals: tuple[str, ...]
) -> list[np.ndarray]:
    """Return metal polygons in port order, rejecting floating or shorted terminals."""
    polygons = component.get_polygons_points(by="tuple")[
        gf.get_layer_tuple(LAYER.M1_DRAW)
    ]
    ordered = []
    indices = []
    for name in terminals:
        point = Point(component.ports[name].center)
        matches = [
            i
            for i, p in enumerate(polygons)
            if Polygon(p).distance(point) <= component.kcl.dbu
        ]
        if len(matches) != 1:
            raise ValueError(f"Terminal {name!r} must touch exactly one metal polygon")
        indices.append(matches[0])
        ordered.append(polygons[matches[0]])
    if len(set(indices)) != len(polygons) or len(indices) != len(polygons):
        raise ValueError("Every metal polygon must have exactly one terminal")
    return ordered


def generate(
    *,
    grid: Mapping[str, Sequence[float]] = GRID,
    output: Path = Path("build/datasets") / NAME,
    workdir: Path = Path("build/palace/plate-capacitor"),
    settings: Settings = SETTINGS,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace",
) -> Dataset:
    """Solve the grid, resume completed matching runs, and publish validated data."""
    PDK.activate()
    executable, files = runtime(workdir, executable, sif, container_binary)
    provenance = {
        **fingerprint(Path(__file__)),
        **files,
        "settings": asdict(settings),
        "processes": processes,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "APPTAINERENV_OMP_NUM_THREADS",
            )
        },
        "metal": "zero-thickness perfect conductor sheets",
        "outer_boundary": "natural zero normal electric displacement",
    }
    metadata = DatasetMetadata(
        name=NAME,
        description="Palace Maxwell capacitance of QPDK plate-capacitor sheets on silicon with a separate coplanar ground frame.",
        axes=tuple(Axis(name=name, unit="um") for name in GRID),
        variants=("cross_section",),
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
        ),
        terminals=TERMINALS,
        reference_ground="coplanar_ground_frame",
        provenance=provenance,
    )

    def solve(**point: float | str) -> dict:
        component = plate_capacitor(**point)
        polygons = [Polygon(p) for p in _terminal_polygons(component, TERMINALS)]
        left = min(p.bounds[0] for p in polygons)
        bottom = min(p.bounds[1] for p in polygons)
        right = max(p.bounds[2] for p in polygons)
        top = max(p.bounds[3] for p in polygons)

        def rectangle(pad: float) -> Polygon:
            return box(left - pad, bottom - pad, right + pad, top + pad)

        sheets = dict(zip(TERMINALS, polygons, strict=True))
        sheets["ground"] = rectangle(settings.ground_pad).difference(
            rectangle(settings.ground_clearance)
        )
        inputs = {
            "point": point,
            "terminals": TERMINALS,
            "polygons_um": [np.asarray(p.exterior.coords).tolist() for p in polygons],
            "provenance": {
                name: value
                for name, value in provenance.items()
                if name != "generator_sha256"
            },
        }
        matrix = extract(
            sheets,
            rectangle(settings.domain_pad),
            height=settings.domain_pad,
            inputs=inputs,
            workdir=workdir,
            executable=executable,
            processes=processes,
            near_mesh=settings.near_mesh,
            far_mesh=settings.far_mesh,
            permittivity=settings.permittivity,
            order=settings.order,
            tolerance=settings.tolerance,
            save_fields=settings.save_fields,
        )
        return {"maxwell_capacitance": matrix}

    logger.info(f"Generating {NAME}: grid={grid}; settings={settings}")
    result = write(
        output, metadata, sweep(metadata, solve, grid, {"cross_section": ["cpw"]})
    )
    result.grid("maxwell_capacitance", cross_section="cpw")
    return result


def main() -> None:
    """Preview or run this experiment; edit the constants for another sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge-shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("build/datasets") / NAME)
    parser.add_argument(
        "--workdir", type=Path, default=Path("build/palace/plate-capacitor")
    )
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--executable", type=Path)
    runtime.add_argument("--sif", type=Path)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--container-binary", default="palace")
    args = parser.parse_args()
    if args.merge_shards is not None:
        merge(
            args.merge_shards,
            args.output,
            grid=GRID,
            variants={"cross_section": ["cpw"]},
        )
        return
    grid = partition_grid(GRID, shard=args.shard, shards=args.shards)
    logger.info(
        f"Grid: {grid}; settings: {SETTINGS}; output: {args.output}; runs: {args.workdir}"
    )
    if not args.dry_run:
        generate(
            grid=grid,
            output=args.output,
            workdir=args.workdir,
            executable=args.executable,
            sif=args.sif,
            processes=args.processes,
            container_binary=args.container_binary,
        )


if __name__ == "__main__":
    main()
