# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/jackgdsf/quantum-rf-pdk.git@a6bdc3e7f4bb38145f3e112c08f1c4dad4dc5b66",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@40d10d86dcef68be92913bb63823b1ea7ac9faf1",
# ]
# ///
"""Generate Maxwell capacitance data with gsim and Palace.

Run ``uv run --script qpdk/models/datasets/data/plate_capacitor.py --help`` from a checkout.
Edit GRID and SETTINGS to define your experiment. The electrodes are perfect
conductor sheets on silicon with a separate coplanar ground frame. Mesh and
domain sensitivity must be checked before relying on small differences.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import gdsfactory as gf
import typer

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


def geometry(
    **point: float | str,
) -> tuple[gf.Component, tuple[float, float, float, float]]:
    """QPDK electrodes inside a separate coplanar ground frame."""
    device = plate_capacitor(**point)
    bounds = device.dbbox(gf.get_layer(LAYER.M1_DRAW))
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    component = gf.Component()
    component.add_ref(device)
    component.add_ports(device.ports)
    return component, (left, bottom, right, top)


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
    from _palace import (  # ruff: ignore[import-outside-top-level]
        extract,
        fingerprint,
        runtime,
    )

    execution, files = runtime(executable, sif, container_binary)
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
            *(
                Quantity(name=name, kind=QuantityKind.CIRCUIT_PARAMETER, unit="1")
                for name in (
                    "fem_error_indicator_norm",
                    "solver_relative_residual",
                    "solver_iterations",
                )
            ),
        ),
        terminals=TERMINALS,
        reference_ground="coplanar_ground_frame",
        provenance=provenance,
    )

    def solve(**point: float | str) -> dict:
        component, (left, bottom, right, top) = geometry(**point)

        def rectangle(pad: float) -> gf.Component:
            result = gf.Component()
            result.add_polygon(
                [
                    (left - pad, bottom - pad),
                    (right + pad, bottom - pad),
                    (right + pad, top + pad),
                    (left - pad, top + pad),
                ],
                layer=LAYER.M1_DRAW,
            )
            return result

        component.add_ref(
            gf.boolean(
                rectangle(settings.ground_pad),
                rectangle(settings.ground_clearance),
                operation="not",
                layer=LAYER.M1_DRAW,
            )
        )
        domain_bounds = (
            left - settings.domain_pad,
            bottom - settings.domain_pad,
            right + settings.domain_pad,
            top + settings.domain_pad,
        )
        inputs = {
            "point": point,
            "terminals": TERMINALS,
            "polygons_um": [
                p.tolist()
                for p in component.get_polygons_points(by="tuple")[
                    gf.get_layer_tuple(LAYER.M1_DRAW)
                ]
            ],
            "provenance": {
                name: value
                for name, value in provenance.items()
                if name != "generator_sha256"
            },
        }
        return extract(
            component,
            domain_bounds,
            {name: name for name in TERMINALS},
            height=settings.domain_pad,
            inputs=inputs,
            workdir=workdir,
            execution=execution,
            processes=processes,
            near_mesh=settings.near_mesh,
            far_mesh=settings.far_mesh,
            permittivity=settings.permittivity,
            order=settings.order,
            tolerance=settings.tolerance,
            save_fields=settings.save_fields,
        )

    logger.info(f"Generating {NAME}: grid={grid}; settings={settings}")
    result = write(
        output, metadata, sweep(metadata, solve, grid, {"cross_section": ["cpw"]})
    )
    result.grid("maxwell_capacitance", cross_section="cpw")
    return result


def main(
    dry_run: Annotated[
        bool, typer.Option(help="Preview the selected grid without running Palace.")
    ] = False,
    shard: Annotated[int, typer.Option(min=0, help="Zero-based worker index.")] = 0,
    shards: Annotated[int, typer.Option(min=1, help="Total number of workers.")] = 1,
    merge_shards: Annotated[
        list[Path] | None,
        typer.Option(help="Completed shard directory; repeat for each shard."),
    ] = None,
    output: Path = Path("build/datasets") / NAME,
    workdir: Path = Path("build/palace/plate-capacitor"),
    executable: Path | None = None,
    sif: Path | None = None,
    processes: Annotated[
        int, typer.Option(min=1, help="MPI ranks per Palace solve.")
    ] = 4,
    container_binary: str = "palace",
) -> None:
    """Generate this experiment, preview it, or merge completed shards."""
    if executable is not None and sif is not None:
        raise typer.BadParameter("Choose --executable or --sif, not both")
    if merge_shards:
        merge(merge_shards, output, grid=GRID, variants={"cross_section": ["cpw"]})
        return
    try:
        grid = partition_grid(GRID, shard=shard, shards=shards)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    logger.info(f"Grid: {grid}; settings: {SETTINGS}; output: {output}")
    if not dry_run:
        generate(
            grid=grid,
            output=output,
            workdir=workdir,
            executable=executable,
            sif=sif,
            processes=processes,
            container_binary=container_binary,
        )


if __name__ == "__main__":
    typer.run(main)
