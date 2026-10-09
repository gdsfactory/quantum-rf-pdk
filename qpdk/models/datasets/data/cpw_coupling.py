# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/jackgdsf/quantum-rf-pdk.git@92b65f5f71e5764e750c0abcf5e1bd12aa10eb53",
#   "typer>=0.24,<1",
#   "gsim @ git+https://github.com/gdsfactory/gsim.git@05c6c93cc14522f8a6a78a08b28116e242cdf1c0",
# ]
# ///
"""Generate symmetric edge-coupled CPW capacitance with meshwell and Palace.

Run ``uv run --script qpdk/models/datasets/data/cpw_coupling.py --help`` from a checkout.
Edit GRID and SETTINGS to define the experiment. Both traces have the same
width and outer slot width; the gap between them is fully etched, with no
intervening ground strip. Perfect conductor sheets lie between air and silicon.

A uniform slice with natural end boundaries eliminates end fringing. Values
are slice capacitances in F; divide by ``slice_length_um * 1e-6`` for F/m.
The dataset-backed SAX model uses the quasi-TEM approximation and excludes
kinetic inductance, conductor loss, finite substrate thickness and dispersion.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
from shapely import Polygon, box, union_all

from qpdk import logger
from qpdk.models.couplers import cpw_cpw_coupling_capacitance_per_length_analytical
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write
from qpdk.tech import material_properties

NAME = "cpw_coupling_palace"
GRID = {
    "width": [
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        20.0,
        25.0,
        30.0,
    ],
    "cpw_gap": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0],
    "gap": [
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        60.0,
        75.0,
        90.0,
        100.0,
        125.0,
        150.0,
        175.0,
        200.0,
        225.0,
        250.0,
    ],
}
TERMINALS = ("lower", "upper")


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Mesh and slice dimensions in µm; substrate relative permittivity."""

    slice_length_um: float = 4.0
    domain_pad: float = 1350.0
    near_mesh: float = 0.14
    far_mesh: float = 30.0
    permittivity: float = material_properties["Si"]["relative_permittivity"]
    order: int = 2
    tolerance: float = 1e-9
    save_fields: bool = False

    def __post_init__(self) -> None:
        """Reject invalid physical or numerical settings."""
        if (
            min(
                self.slice_length_um, self.domain_pad, self.near_mesh, self.permittivity
            )
            <= 0
        ):
            raise ValueError("Slice, domain, mesh and permittivity must be positive")
        if (
            self.near_mesh > self.far_mesh
            or self.order < 1
            or not 0 < self.tolerance < 1
        ):
            raise ValueError(
                "Require near_mesh <= far_mesh, order >= 1 and 0 < tolerance < 1"
            )


SETTINGS = Settings()


def geometry(
    width: float, cpw_gap: float, gap: float, settings: Settings
) -> tuple[dict[str, Polygon], Polygon]:
    """Two identical traces and outer ground rails spanning the whole slice."""
    if min(width, cpw_gap, gap) <= 0:
        raise ValueError("Trace width, outer slot and inter-trace gap must be positive")
    length = settings.slice_length_um
    inner = gap / 2
    edge = inner + width
    extent = edge + cpw_gap + settings.domain_pad
    sheets = {
        "lower": box(0, -edge, length, -inner),
        "upper": box(0, inner, length, edge),
        "ground": union_all([
            box(0, -extent, length, -edge - cpw_gap),
            box(0, edge + cpw_gap, length, extent),
        ]),
    }
    return sheets, box(0, -extent, length, extent)


def generate(
    *,
    grid: Mapping[str, Sequence[float]] = GRID,
    output: Path = Path("build/datasets") / NAME,
    workdir: Path = Path("build/palace/cpw-coupling"),
    settings: Settings = SETTINGS,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace",
) -> Dataset:
    """Solve every geometry, resume matching results and publish complete data."""
    from _palace import (  # ruff: ignore[import-outside-top-level]
        extract,
        fingerprint,
        runtime,
    )

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
        "topology": "symmetric edge-coupled CPW; fully etched gap between traces",
        "outer_boundary": "natural zero normal electric displacement, including both slice ends",
        "slice_length_um": settings.slice_length_um,
        "analytical_reference": "Conformal mapping for symmetric edge-coupled CPW dielectric half-spaces; relative difference = FEM mutual / analytical mutual - 1",
    }
    metadata = DatasetMetadata(
        name=NAME,
        description="Palace Maxwell capacitance of a uniform symmetric edge-coupled CPW slice on silicon.",
        axes=tuple(Axis(name=name, unit="um") for name in GRID),
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
                    "analytical_mutual_relative_difference",
                )
            ),
        ),
        terminals=TERMINALS,
        reference_ground="outer_coplanar_ground_rails",
        provenance=provenance,
    )

    def solve(**point: float) -> dict:
        sheets, footprint = geometry(**point, settings=settings)
        inputs = {
            "point": point,
            "terminals": TERMINALS,
            "provenance": {
                key: value
                for key, value in provenance.items()
                if key != "generator_sha256"
            },
        }
        result = extract(
            sheets,
            footprint,
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
        reference = (
            float(
                cpw_cpw_coupling_capacitance_per_length_analytical(
                    **point, ep_r=settings.permittivity
                )
            )
            * settings.slice_length_um
            * 1e-6
        )
        result["analytical_mutual_relative_difference"] = (
            -result["maxwell_capacitance"][0, 1] / reference - 1
        )
        return result

    logger.info(f"Generating {NAME}: grid={grid}; settings={settings}")
    result = write(output, metadata, sweep(metadata, solve, grid))
    result.grid("maxwell_capacitance")
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
    workdir: Path = Path("build/palace/cpw-coupling"),
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
        merge(merge_shards, output, grid=GRID, variants=None)
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
