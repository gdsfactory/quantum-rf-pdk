"""Shared execution for small component capacitance sweeps.

Recipes supply their geometry, grid and physical conventions. Select mesh
settings with representative refinement checks before generating the full grid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import gdsfactory as gf
import typer

from qpdk import PDK, logger
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.capacitance import maxwell_to_mutual
from qpdk.models.datasets.generate import merge, partition_grid, sweep, write
from qpdk.tech import LAYER, material_properties

if TYPE_CHECKING:
    from gsim.palace import ElectrostaticSim


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Mesh lengths in micrometres and static solver controls."""

    near_mesh: float = 3.0
    far_mesh: float = 100.0
    domain_pad: float = 600.0
    permittivity: float = material_properties["Si"]["relative_permittivity"]
    order: int = 2
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Reject invalid physical dimensions and solver settings."""
        if not 0 < self.near_mesh <= self.far_mesh or self.domain_pad <= 0:
            raise ValueError("Require positive domain and 0 < near_mesh <= far_mesh")
        if self.permittivity <= 0 or self.order < 1 or not 0 < self.tolerance < 1:
            raise ValueError("Invalid permittivity, element order or solver tolerance")


@dataclass(slots=True, kw_only=True)
class Geometry:
    """Resolved electrodes and domain; custom meshes record their inputs in mesh_inputs."""

    component: gf.Component
    bounds: tuple[float, float, float, float]
    terminal_ports: dict[str, str]
    mesh: Callable[[ElectrostaticSim], None] | None = None
    mesh_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Recipe:
    """One component experiment; axis units default to um unless axis_units overrides them."""

    name: str
    script: Path
    grid: dict[str, list[float]]
    terminals: tuple[str, ...]
    geometry: Callable[..., Geometry]
    description: str
    provenance: dict[str, Any]
    settings: Settings = Settings()
    reference_ground: str = "chip_ground"
    variants: dict[str, list[str]] = field(default_factory=dict)
    axis_units: dict[str, str] = field(default_factory=dict)


def generate(
    recipe: Recipe,
    *,
    grid: dict[str, list[float]],
    output: Path,
    workdir: Path,
    executable: Path | None = None,
    sif: Path | None = None,
    processes: int = 4,
    container_binary: str = "palace-x86_64.bin",
    settings: Settings | None = None,
) -> Dataset:
    """Run the existing resumable Palace extraction and Parquet writer."""
    from _palace import (  # ruff: ignore[import-outside-top-level]
        extract,
        fingerprint,
        runtime,
    )

    PDK.activate()
    settings = settings or recipe.settings
    execution, files = runtime(executable, sif, container_binary)
    provenance = {
        **fingerprint(recipe.script),
        **files,
        **recipe.provenance,
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": asdict(settings),
        "processes": processes,
    }
    metadata = DatasetMetadata(
        name=recipe.name,
        description=recipe.description,
        axes=tuple(
            Axis(name=name, unit=recipe.axis_units.get(name, "um"))
            for name in recipe.grid
        ),
        variants=tuple(recipe.variants),
        terminals=recipe.terminals,
        reference_ground=recipe.reference_ground,
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="mutual_capacitance",
                kind=QuantityKind.MUTUAL_CAPACITANCE,
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
        provenance=provenance,
    )

    def solve(**point: float | str) -> dict:
        geometry = recipe.geometry(point, settings, processes)
        if tuple(geometry.terminal_ports) != recipe.terminals:
            raise ValueError("Geometry terminal order differs from the recipe")
        if geometry.mesh is not None and not geometry.mesh_inputs:
            raise ValueError(
                "Custom meshes must record their geometry and meshing inputs"
            )
        inputs = {
            "point": point,
            "mesh_inputs": geometry.mesh_inputs,
            "bounds_um": geometry.bounds,
            "terminals": {
                name: list(geometry.component.ports[port].center)
                for name, port in geometry.terminal_ports.items()
            },
            "polygons_um": {
                str(layer): [p.tolist() for p in polygons]
                for layer, polygons in geometry.component.get_polygons_points(
                    by="tuple"
                ).items()
            },
            "provenance": {
                name: value
                for name, value in provenance.items()
                if name not in {"generator_sha256", "experiment_sha256"}
            },
        }
        result = extract(
            geometry.component,
            geometry.bounds,
            geometry.terminal_ports,
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
            mesh=geometry.mesh,
        )
        return {
            **result,
            "mutual_capacitance": maxwell_to_mutual(result["maxwell_capacitance"]),
        }

    result = write(output, metadata, sweep(metadata, solve, grid, recipe.variants))
    for selection in product(*recipe.variants.values()):
        result.grid(
            "maxwell_capacitance",
            allow_missing=False,
            **dict(zip(recipe.variants, selection, strict=True)),
        )
    return result


def run_cli(recipe: Recipe) -> None:
    """Use the same preview, shard and merge options as the existing generators."""

    def main(
        dry_run: bool = False,
        shard: Annotated[int, typer.Option(min=0)] = 0,
        shards: Annotated[int, typer.Option(min=1)] = 1,
        merge_shards: Annotated[list[Path] | None, typer.Option()] = None,
        output: Path | None = None,
        workdir: Path | None = None,
        executable: Path | None = None,
        sif: Path | None = None,
        processes: Annotated[int, typer.Option(min=1)] = 4,
        container_binary: str = "palace-x86_64.bin",
    ) -> None:
        """Preview, generate or merge this component dataset."""
        if executable is not None and sif is not None:
            raise typer.BadParameter("Choose --executable or --sif")
        output = output or Path("build/datasets") / recipe.name
        workdir = workdir or Path("build/palace") / recipe.name
        if merge_shards:
            merge(merge_shards, output, grid=recipe.grid, variants=recipe.variants)
            return
        grid = partition_grid(recipe.grid, shard=shard, shards=shards)
        count = prod(map(len, grid.values())) * prod(map(len, recipe.variants.values()))
        logger.info(
            f"{recipe.name}: {count} points; grid={grid}; settings={recipe.settings}"
        )
        if not dry_run:
            generate(
                recipe,
                grid=grid,
                output=output,
                workdir=workdir,
                executable=executable,
                sif=sif,
                processes=processes,
                container_binary=container_binary,
            )

    typer.run(main)


def ground_frame(
    component: gf.Component, *, clearance: float, width: float, domain_pad: float
) -> tuple[float, float, float, float]:
    """Add a finite coplanar frame without changing the device electrodes."""
    box = component.dbbox(gf.get_layer(LAYER.M1_DRAW))

    def rectangle(pad: float) -> gf.Component:
        shape = gf.Component()
        shape.add_polygon(
            [
                (box.left - pad, box.bottom - pad),
                (box.right + pad, box.bottom - pad),
                (box.right + pad, box.top + pad),
                (box.left - pad, box.top + pad),
            ],
            layer=LAYER.M1_DRAW,
        )
        return shape

    component.add_ref(
        gf.boolean(
            rectangle(clearance + width),
            rectangle(clearance),
            operation="not",
            layer=LAYER.M1_DRAW,
        )
    )
    return (
        box.left - clearance - width - domain_pad,
        box.bottom - clearance - width - domain_pad,
        box.right + clearance + width + domain_pad,
        box.top + clearance + width + domain_pad,
    )


def stack_geometry(
    electrodes: dict[str, tuple[gf.kdb.Region, float, float]],
    *,
    terminals: tuple[str, ...],
    dielectrics: list[dict],
    settings: Settings,
    processes: int,
) -> Geometry:
    """Assign disconnected electrodes to extraction layers in a finite gsim stack."""
    component = gf.Component()
    layers = {}
    # gsim selects terminals by layer, so disconnected electrodes need separate layers.
    for i, (name, (region, bottom, top)) in enumerate(electrodes.items()):
        if region.is_empty() or not top > bottom:
            raise ValueError(
                f"Electrode {name!r} needs a nonempty region and top > bottom"
            )
        layer = (900 + i, 0)
        for polygon in region.merged().each():
            component.add_polygon(polygon.to_dtype(component.kcl.dbu), layer=layer)
        if name in terminals:
            polygon = next(region.each())
            vertex = next(polygon.each_point_hull())
            component.add_port(
                name,
                center=(vertex.x * component.kcl.dbu, vertex.y * component.kcl.dbu),
                width=1,
                orientation=180,
                layer=layer,
                port_type="electrical",
            )
        layers[name] = dict(
            name=name,
            gds_layer=layer,
            zmin=bottom,
            zmax=top,
            thickness=top - bottom,
            material="pec",
            layer_type="conductor",
        )
    stack = dict(
        pdk_name="qpdk-extraction",
        layers=layers,
        materials={
            "silicon": {"permittivity": settings.permittivity},
            "air": {"permittivity": 1.0},
            "pec": {"conductivity": 1e30},
        },
        dielectrics=dielectrics,
    )

    airbox = dict(
        margin_x=settings.domain_pad,
        margin_y=settings.domain_pad,
        z_above=settings.domain_pad,
        z_below=settings.domain_pad,
    )
    mesh_options = dict(
        refined_mesh_size=settings.near_mesh,
        max_mesh_size=settings.far_mesh,
        margin=0,
        planar_conductors=False,
        threads=processes,
        surface_threads=1,
        verbose=True,
    )

    def mesh(sim: ElectrostaticSim) -> None:
        """Use the public gsim stack mesher and electrostatic terminals."""
        from gsim.common import LayerStack  # ruff: ignore[import-outside-top-level]

        sim.set_stack(LayerStack.model_validate(stack))
        for name in terminals:
            sim.add_terminal(name, layer=name)
        sim.set_airbox(**airbox)
        sim.mesh(**mesh_options)

    bounds = component.dbbox()
    pad = settings.domain_pad
    return Geometry(
        component=component,
        bounds=(
            bounds.left - pad,
            bounds.bottom - pad,
            bounds.right + pad,
            bounds.top + pad,
        ),
        terminal_ports={name: name for name in terminals},
        mesh=mesh,
        mesh_inputs={
            "stack": stack,
            "airbox": airbox,
            "mesh": mesh_options,
        },
    )
