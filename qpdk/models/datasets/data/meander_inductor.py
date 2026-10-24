# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@c00764bc18cbcb0e58ccb5c7dfb880d550455d35",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Generate two-port meander S-parameters with a driven Palace solve.

The winding lies between air and silicon, inside a coplanar ground frame
100 um wide and 20 um from its bounding box. Each 50 ohm lumped port spans
the gap from one wire end to the frame. The reference planes are the wire
ends, with no added feedlines or de-embedding. Both ports are excited
independently. The frame is part of the device environment.

The thin film has a fixed sheet kinetic inductance of 0.4 pH per square,
including the ground frame. This is a material assumption, not a measured
film property. Geometric inductance and distributed capacitance come from
the field solve. Dielectric and conductor loss are omitted, while the outer
boundary absorbs radiation. The impedance convention follows
https://awslabs.github.io/palace/stable/guide/boundaries/#Impedance-boundary.

Use ``s_parameters_model(path, n_turns="5")`` to interpolate the complex
matrix in frequency and geometry. Turn count is a discrete selection.
The equipotential diagnostic remains in ``meander_ground_capacitance.py``.

Preview with ``uv run --script qpdk/models/datasets/data/meander_inductor.py --dry-run``.
Check mesh, element order, domain and held-out frequencies before generating
the full grid. Each geometry solves the whole frequency sweep once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import jax.numpy as jnp
from _driven import Recipe, run_cli
from _experiment import ground_frame

from qpdk.cells import meander_inductor
from qpdk.tech import LAYER, coplanar_waveguide, material_properties

if TYPE_CHECKING:
    from gsim.palace import DrivenSim

GRID = {
    "turn_length": [100.0, 200.0, 300.0],
    "wire_width": [1.0, 2.0],
    "wire_gap": [2.0, 4.0, 8.0],
}
VARIANTS = {"n_turns": ["3", "5", "7"]}
NAME = "meander_sparameters_palace"
TERMINALS = ("o1", "o2")
CLEARANCE = 20.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Lengths in um, frequencies in Hz, sheet inductance in H per square."""

    near_mesh: float = 0.5
    far_mesh: float = 100.0
    domain_pad: float = 300.0
    order: int = 2
    tolerance: float = 1e-8
    sheet_inductance: float = 0.4e-12
    fmin: float = 0.5e9
    fmax: float = 20e9
    points: int = 40

    def __post_init__(self) -> None:
        """Reject invalid mesh and material inputs before starting a solve."""
        if not 0 < self.near_mesh <= self.far_mesh or self.domain_pad <= 0:
            raise ValueError("Require positive domain and 0 < near_mesh <= far_mesh")
        if self.sheet_inductance < 0 or self.order < 1 or not 0 < self.tolerance < 1:
            raise ValueError("Invalid film or solver settings")
        if not 0 < self.fmin < self.fmax or self.points < 2:
            raise ValueError(
                "Require positive increasing frequencies and at least two points"
            )


SETTINGS = Settings()


def geometry(point: dict[str, float | str]) -> gf.Component:
    """Reuse the winding cell and add the specified return conductor."""
    if int(point["n_turns"]) < 1 or min(float(point[name]) for name in GRID) <= 0:
        raise ValueError("Turn count and winding dimensions must be positive")
    device = meander_inductor(
        n_turns=int(point["n_turns"]),
        turn_length=float(point["turn_length"]),
        wire_gap=float(point["wire_gap"]),
        cross_section=coplanar_waveguide(width=float(point["wire_width"]), gap=2.0),
    )
    component = gf.Component()
    component.add_ref(device)
    component.add_ports(device.ports)
    ground_frame(component, clearance=CLEARANCE, width=100.0, domain_pad=0.0)
    return component


def simulation(
    point: dict[str, float | str], settings: Settings, output: Path, processes: int
) -> DrivenSim:
    """Set up a thin-film driven solve without a volume mesh inside the metal."""
    from gsim.common import Layer, LayerStack  # ruff: ignore[import-outside-top-level]
    from gsim.palace import DrivenSim  # ruff: ignore[import-outside-top-level]

    component = geometry(point)
    sim = DrivenSim()
    sim.set_geometry(component)
    sim.set_stack(
        LayerStack(
            layers={
                "metal": Layer(
                    name="metal",
                    gds_layer=gf.get_layer_tuple(LAYER.M1_DRAW),
                    zmin=0.0,
                    zmax=0.2,
                    thickness=0.2,
                    material="aluminum",
                    layer_type="conductor",
                )
            },
            dielectrics=[
                dict(
                    name="substrate",
                    zmin=-settings.domain_pad,
                    zmax=0.0,
                    material="coupon_silicon",
                ),
                dict(name="air", zmin=0.0, zmax=settings.domain_pad, material="air"),
            ],
            materials={
                "coupon_silicon": dict(
                    permittivity=material_properties["Si"]["relative_permittivity"],
                    permeability=1.0,
                    loss_tangent=0.0,
                    conductivity=0.0,
                ),
                "air": dict(permittivity=1.0, permeability=1.0),
            },
        )
    )
    sim.set_airbox(
        margin_x=settings.domain_pad,
        margin_y=settings.domain_pad,
        z_above=settings.domain_pad,
        z_below=settings.domain_pad,
    )
    for name in TERMINALS:
        # Inplane ports are centered on their sheet, halfway across the gap.
        sim.add_port(
            name,
            layer="metal",
            length=CLEARANCE,
            offset=-CLEARANCE / 2,
            impedance=50.0,
            excited=True,
        )
    sim.set_output_dir(output)
    sim.set_solver(
        order=settings.order,
        tolerance=settings.tolerance,
        max_iterations=1000,
        preconditioner="AMS",
        initial_guess=False,
    )
    sim.set_driven(
        fmin=settings.fmin,
        fmax=settings.fmax,
        num_points=settings.points,
        adaptive_tol=0.0,
        save_step=0,
    )
    sim.mesh(
        refined_mesh_size=min(settings.near_mesh, float(point["wire_width"]) / 2),
        max_mesh_size=settings.far_mesh,
        planar_conductors=True,
        threads=processes,
        surface_threads=1,
        verbose=False,
    )
    tetrahedra = sim._last_mesh_result.mesh_stats["tetrahedra"]
    if tetrahedra > (800_000 if settings.order >= 3 else 1_500_000):
        raise ValueError(f"Calibration mesh budget exceeded: {tetrahedra} tetrahedra")
    return sim


def write_config(sim: DrivenSim, settings: Settings) -> Path:
    """Apply the film's per-square inductance to the planar metal attributes."""
    path = sim.write_config()
    config = json.loads(path.read_text())
    boundaries = config["Boundaries"]
    if boundaries.get("Conductivity") or not boundaries.get("Absorbing"):
        raise ValueError("Expected planar metal and an absorbing outer boundary")
    if len(boundaries["LumpedPort"]) != 2 or {
        p["Excitation"] for p in boundaries["LumpedPort"]
    } != {1, 2}:
        raise ValueError("Both ports must be excited independently")
    if settings.sheet_inductance > 0:
        attributes = boundaries.pop("PEC")["Attributes"]
        boundaries["Impedance"] = [
            {"Attributes": attributes, "Ls": settings.sheet_inductance}
        ]
    path.write_text(json.dumps(config, indent=2))
    return path


def prepare(point: dict[str, float | str], output: Path, processes: int) -> DrivenSim:
    """Prepare the meshed winding with its film impedance boundary."""
    sim = simulation(point, SETTINGS, output, processes)
    write_config(sim, SETTINGS)
    return sim


RECIPE = Recipe(
    name=NAME,
    script=Path(__file__),
    grid=GRID,
    frequencies=[
        float(f) for f in jnp.linspace(SETTINGS.fmin, SETTINGS.fmax, SETTINGS.points)
    ],
    variants=VARIANTS,
    terminals=TERMINALS,
    prepare=prepare,
    description="Driven meander S-parameters with a fixed kinetic sheet inductance and finite coplanar ground",
    reference_ground="coplanar_return_frame",
    provenance={
        "settings": asdict(SETTINGS),
        "reference_impedance_ohm": 50.0,
        "reference_planes": "Wire ends, each excited across the 20 um gap to the ground frame, without de-embedding",
        "ground": "100 um wide coplanar frame, 20 um from the winding bounding box",
        "film": "Uniform lossless kinetic sheet inductance on winding and ground; geometric inductance from the field solve",
        "validation": "Mesh, order, domain and interpolation checks are required before curation",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
