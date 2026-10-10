# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/jackgdsf/quantum-rf-pdk.git@a6bdc3e7f4bb38145f3e112c08f1c4dad4dc5b66",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@8e9b8ff1deba37e575217b99cfa556ea8060a8e5",
# ]
# ///
"""Generate symmetric edge-coupled CPW capacitance with meshwell and Palace.

Run ``uv run --script qpdk/models/datasets/data/cpw_coupling.py --help`` from a checkout.
Edit GRID and SETTINGS to define the experiment. Both traces have the same
width and outer slot width. ``--topology as-drawn`` retains the ground strip
between separated CPW slots, matching ``coupler_straight``. The default
``fully-etched`` gap supplies a comparison with the analytical ECCPW formula.
Perfect conductor sheets lie between air and silicon.

The grounded experiment samples ``ground_strip_width = gap - 2 * cpw_gap``
on a logarithmic grid. Its validated strip widths start at 0.01 µm; narrower
positive strips require additional simulations. Logarithmic interpolation of
coordinates and capacitance magnitudes resolves the rapid shielding onset.

A two-dimensional transverse mesh eliminates end fringing and longitudinal
mesh overhead. Palace's ``Model.Lc`` sets the implicit depth to
``slice_length_um * 1e-6``. Values are slice capacitances in F; divide by this
depth for F/m. The physical result is independent of that normalization depth.
The dataset-backed SAX model uses the quasi-TEM approximation and excludes
kinetic inductance, conductor loss, finite substrate thickness and dispersion.

Each geometry is solved on successively finer meshes until every raw matrix
entry changes by at most 1% and the equal traces agree within 1%. The accepted
finer result records its mesh sizes, refinement level and measured change.
Domain truncation and interpolation need separate checks.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import gdsfactory as gf
import jax.numpy as jnp
import typer
from jax.typing import ArrayLike

from qpdk import PDK, logger
from qpdk.cells import coupler_straight
from qpdk.models.couplers import cpw_cpw_coupling_capacitance_per_length_analytical
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write
from qpdk.tech import LAYER, coplanar_waveguide, material_properties

NAME = "cpw_coupling_palace"
GROUND_STRIP_NAME = "cpw_coupling_ground_strip_palace"
GRID = {
    "width": [2.0, 6.0, 10.0, 30.0],
    "cpw_gap": [1.0, 3.0, 6.0, 20.0],
    "gap": [
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        25.0,
        50.0,
        75.0,
        100.0,
        175.0,
        250.0,
    ],
}
TERMINALS = ("lower", "upper")
GROUND_GRID = {
    "width": [2.0, 3.0, 6.0, 10.0, 20.0, 30.0],
    "cpw_gap": [1.0, 2.0, 3.0, 6.0, 10.0, 20.0],
    "ground_strip_width": [
        0.01,
        0.033,
        0.11,
        0.36,
        1.2,
        4.0,
        9.0,
        13.0,
        24.0,
        54.0,
        128.0,
        248.0,
    ],
}


class Topology(StrEnum):
    """Ground left between two CPW etch masks, or a fully etched inner gap."""

    FULLY_ETCHED = "fully-etched"
    AS_DRAWN = "as-drawn"


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
GROUND_SETTINGS = replace(SETTINGS, domain_pad=2025.0)


def mesh_settings(settings: Settings, refinements: int) -> tuple[Settings, ...]:
    """Refine near-conductor and far-field sizes together."""
    return (
        settings,
        *(
            replace(
                settings,
                near_mesh=round(settings.near_mesh * (0.1 / 0.14) * 0.7**level, 12),
                far_mesh=round(settings.far_mesh * (20 / 30) * 0.7**level, 12),
            )
            for level in range(refinements)
        ),
    )


def converge_mesh(
    solve: Callable[[Settings], dict[str, ArrayLike]],
    settings: Settings,
    *,
    tolerance: float = 0.01,
    max_refinements: int = 4,
) -> dict[str, ArrayLike]:
    """Accept a finer solve only after every matrix entry changes by at most tolerance.

    Compare raw entries, including small mutual capacitances. The equal traces
    must also agree within 1%. This measures successive-mesh sensitivity, not
    a rigorous error bound or domain/interpolation convergence.

    Returns:
        Accepted solver quantities and measured mesh diagnostics.

    Raises:
        ValueError: If the tolerance or refinement limit is invalid.
        RuntimeError: If a matrix is invalid or the refinement limit is exhausted.
    """
    if not 0 < tolerance < 1 or max_refinements < 1:
        raise ValueError("Require 0 < mesh tolerance < 1 and at least one refinement")
    previous = None
    change = float("inf")
    for level, config in enumerate(mesh_settings(settings, max_refinements)):
        result = solve(config)
        matrix = jnp.asarray(result["maxwell_capacitance"])
        if matrix.shape != (2, 2) or not bool(jnp.all(jnp.isfinite(matrix))):
            raise RuntimeError(
                "Mesh convergence requires a finite 2x2 capacitance matrix"
            )
        if previous is not None:
            change = float(jnp.max(jnp.abs((matrix - previous) / matrix)))
            symmetric = bool(jnp.isclose(matrix[0, 0], matrix[1, 1], rtol=0.01, atol=0))
            logger.info(
                f"Mesh refinement {level}: maximum relative change={change:.3%}"
            )
            if change <= tolerance and symmetric:
                return {
                    **result,
                    "mesh_relative_change": change,
                    "mesh_refinement_level": float(level),
                    "mesh_near_size": config.near_mesh * 1e-6,
                    "mesh_far_size": config.far_mesh * 1e-6,
                }
        previous = matrix
    raise RuntimeError(
        f"Capacitance did not converge after {max_refinements} mesh refinements; "
        f"last relative change={change:.3%}"
    )


def geometry(
    width: float,
    cpw_gap: float,
    gap: float,
    settings: Settings,
    *,
    topology: Topology = Topology.FULLY_ETCHED,
) -> tuple[gf.Component, tuple[float, float, float, float]]:
    """Mesh the QPDK coupler layout, optionally etching the whole inner gap."""
    if min(width, cpw_gap, gap) <= 0:
        raise ValueError("Trace width, outer slot and inter-trace gap must be positive")
    PDK.activate()
    length = settings.slice_length_um
    extent = gap / 2 + width + cpw_gap + settings.domain_pad
    device = gf.Component()
    reference = device.add_ref(
        coupler_straight(
            length=length,
            gap=gap,
            cross_section=coplanar_waveguide(width=width, gap=cpw_gap),
        )
    )
    reference.dmovey(-(width + gap) / 2)
    slots = device.extract([LAYER.M1_ETCH])
    if topology == Topology.FULLY_ETCHED:
        slots.add_polygon(
            [(0, -gap / 2), (length, -gap / 2), (length, gap / 2), (0, gap / 2)],
            layer=LAYER.M1_ETCH,
        )
    background = gf.Component()
    background.add_polygon(
        [(0, -extent), (length, -extent), (length, extent), (0, extent)],
        layer=LAYER.M1_DRAW,
    )
    component = gf.Component()
    component.add_ref(
        gf.boolean(
            background,
            slots,
            operation="not",
            layer=LAYER.M1_DRAW,
            layer1=LAYER.M1_DRAW,
            layer2=LAYER.M1_ETCH,
        )
    )
    # Preserve signal metal where the neighbouring CPW slot overlaps it.
    component.add_ref(device.extract([LAYER.M1_DRAW]))
    left_ports = sorted(
        reference.ports, key=lambda port: (port.center[0], port.center[1])
    )[:2]
    for name, port in zip(TERMINALS, left_ports, strict=True):
        component.add_port(name=name, port=port, port_type="electrical")
    return component, (0, -extent, length, extent)


def generate(
    *,
    grid: Mapping[str, Sequence[float]] | None = None,
    output: Path | None = None,
    workdir: Path = Path("build/palace/cpw-coupling"),
    settings: Settings | None = None,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace",
    mesh_tolerance: float = 0.01,
    max_refinements: int = 4,
    topology: Topology = Topology.FULLY_ETCHED,
) -> Dataset:
    """Solve every geometry, resume matching results and publish complete data."""
    from _palace import (  # ruff: ignore[import-outside-top-level]
        extract,
        fingerprint,
        runtime,
    )

    if not 0 < mesh_tolerance < 1 or max_refinements < 1:
        raise ValueError("Require 0 < mesh tolerance < 1 and at least one refinement")
    grid = grid if grid is not None else experiment_grid(topology)
    settings = settings or (
        GROUND_SETTINGS if topology == Topology.AS_DRAWN else SETTINGS
    )
    name = GROUND_STRIP_NAME if topology == Topology.AS_DRAWN else NAME
    output = output or Path("build/datasets") / name
    execution, files = runtime(executable, sif, container_binary)
    provenance = {
        **fingerprint(Path(__file__)),
        **files,
        "settings": asdict(settings),
        "processes": processes,
        "mesh_convergence": {
            "relative_tolerance": mesh_tolerance,
            "criterion": "Maximum relative change of every raw Maxwell matrix entry between successive meshes; equal-trace self-capacitances must also agree within 1%",
            "settings_by_level": [
                asdict(config) for config in mesh_settings(settings, max_refinements)
            ],
            "limitations": "Successive-mesh sensitivity is not a rigorous error bound; domain and interpolation accuracy require separate checks",
        },
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
        "topology": topology.value,
        "geometry_coordinates": "gap = 2 * cpw_gap + ground_strip_width"
        if "ground_strip_width" in grid
        else "gap between conductor edges",
        "interpolation": "logarithmic coordinates and signed capacitance magnitudes"
        if "ground_strip_width" in grid
        else "multilinear",
        "inner_ground": "Ground strip of width max(gap - 2 * cpw_gap, 0) between CPW slots"
        if topology == Topology.AS_DRAWN
        else "Fully etched gap between traces",
        "mesh_dimension": 2,
        "normalization": "Model.Lc equals slice_length_um; 2D capacitance is reported in F for that implicit depth",
        "outer_boundary": "natural zero normal electric displacement at the transverse domain boundary",
        "slice_length_um": settings.slice_length_um,
        "analytical_reference": "Conformal mapping for an unshielded symmetric edge-coupled CPW with dielectric half-spaces; relative difference = FEM mutual / unshielded analytical mutual - 1. With an inner ground strip this measures shielding as well as numerical differences",
    }
    metadata = DatasetMetadata(
        name=name,
        description="Palace Maxwell capacitance of a uniform symmetric edge-coupled CPW slice on silicon.",
        axes=tuple(Axis(name=name, unit="um") for name in grid),
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
                    "mesh_relative_change",
                    "mesh_refinement_level",
                )
            ),
            *(
                Quantity(name=name, kind=QuantityKind.CIRCUIT_PARAMETER, unit="m")
                for name in ("mesh_near_size", "mesh_far_size")
            ),
        ),
        terminals=TERMINALS,
        reference_ground="outer_rails_and_inner_ground_strip"
        if topology == Topology.AS_DRAWN
        else "outer_coplanar_ground_rails",
        provenance=provenance,
    )

    def solve_once(point: dict[str, float], config: Settings) -> dict[str, ArrayLike]:
        component, domain_bounds = geometry(**point, settings=config, topology=topology)
        inputs = {
            "point": point,
            "polygons_um": [
                p.tolist()
                for p in component.get_polygons_points(by="tuple")[
                    gf.get_layer_tuple(LAYER.M1_DRAW)
                ]
            ],
            "terminals": TERMINALS,
            "provenance": {
                key: value
                for key, value in provenance.items()
                if key not in {"generator_sha256", "mesh_convergence"}
            }
            | {"settings": asdict(config)},
        }
        result = extract(
            component,
            domain_bounds,
            {name: name for name in TERMINALS},
            height=config.domain_pad,
            inputs=inputs,
            workdir=workdir,
            execution=execution,
            processes=processes,
            near_mesh=config.near_mesh,
            far_mesh=config.far_mesh,
            permittivity=config.permittivity,
            order=config.order,
            tolerance=config.tolerance,
            save_fields=config.save_fields,
            normalization_depth_um=config.slice_length_um,
            minimum_feature_elements=4 * settings.near_mesh / config.near_mesh,
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

    def solve(**point: float) -> dict[str, ArrayLike]:
        if "ground_strip_width" in point:
            strip = point.pop("ground_strip_width")
            if strip <= 0:
                raise ValueError("Ground-strip width must be positive")
            point["gap"] = 2 * point["cpw_gap"] + strip
        return converge_mesh(
            lambda config: solve_once(point, config),
            settings,
            tolerance=mesh_tolerance,
            max_refinements=max_refinements,
        )

    logger.info(f"Generating {name}: grid={grid}; settings={settings}")
    result = write(output, metadata, sweep(metadata, solve, grid))
    result.grid("maxwell_capacitance")
    return result


def experiment_grid(topology: Topology) -> dict[str, list[float]]:
    """Keep the shielding transition on its own strip-width axis."""
    return GROUND_GRID if topology == Topology.AS_DRAWN else GRID


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
    output: Path | None = None,
    workdir: Path = Path("build/palace/cpw-coupling"),
    executable: Path | None = None,
    sif: Path | None = None,
    processes: Annotated[
        int, typer.Option(min=1, help="MPI ranks per Palace solve.")
    ] = 4,
    container_binary: str = "palace",
    mesh_tolerance: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="Maximum relative change between successive meshes."
        ),
    ] = 0.01,
    max_refinements: Annotated[
        int,
        typer.Option(
            min=1,
            help="Fail rather than publish if this many refinements do not converge.",
        ),
    ] = 4,
    topology: Annotated[
        Topology,
        typer.Option(help="Use as-drawn for the ground left by coupler_straight."),
    ] = Topology.FULLY_ETCHED,
) -> None:
    """Generate this experiment, preview it, or merge completed shards."""
    if executable is not None and sif is not None:
        raise typer.BadParameter("Choose --executable or --sif, not both")
    name = GROUND_STRIP_NAME if topology == Topology.AS_DRAWN else NAME
    output = output or Path("build/datasets") / name
    complete_grid = experiment_grid(topology)
    if merge_shards:
        merge(merge_shards, output, grid=complete_grid, variants=None)
        return
    try:
        grid = partition_grid(complete_grid, shard=shard, shards=shards)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    settings = GROUND_SETTINGS if topology == Topology.AS_DRAWN else SETTINGS
    logger.info(f"Grid: {grid}; settings: {settings}; output: {output}")
    if not dry_run:
        generate(
            grid=grid,
            settings=settings,
            output=output,
            workdir=workdir,
            executable=executable,
            sif=sif,
            processes=processes,
            container_binary=container_binary,
            mesh_tolerance=mesh_tolerance,
            max_refinements=max_refinements,
            topology=topology,
        )


if __name__ == "__main__":
    typer.run(main)
