# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@5dbc4a0d7ccb50f777191575383f689b53555305",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Full-layout electrostatic coupon with open feedline reference cuts.

The readout conductor has one potential, not a shorted quarter-wave mode.
Do not add this full coupon admittance to an existing distributed resonator
model, which already contains its route and ground capacitance. It requires
driven/eigenmode validation before use as an RF coupling model.
The sampled feedline gaps retain a ground strip between the coupled lines.

Edit GRID and SETTINGS, then preview with --dry-run before a new sweep.
The Parquet output retains all Maxwell and mutual branches and diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import TERMINALS, geometry as readout_geometry

from qpdk.tech import LAYER

if TYPE_CHECKING:
    from gsim.palace import ElectrostaticSim

GRID = {
    "pad_length": [300.0, 394.0],
    "pad_offset": [35.0, 55.0],
    "feedline_gap": [14.0, 15.0, 16.0, 18.0, 20.0],
    "coupling_length": [500.0, 800.0],
}
SETTINGS = Settings(near_mesh=1.4, far_mesh=108.0, domain_pad=2000.0, order=3)


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Use the calibrated edge grading with serial, reproducible meshing."""
    result = readout_geometry(point, settings, 1)

    def mesh(sim: ElectrostaticSim) -> None:
        generated = sim.mesh_sheets(
            conductor_layer=gf.get_layer_tuple(LAYER.M1_DRAW),
            terminal_ports=result.terminal_ports,
            domain_bounds=result.bounds,
            height=settings.domain_pad,
            near_mesh=settings.near_mesh,
            far_mesh=settings.far_mesh,
            permittivity=settings.permittivity,
            edge_transition=100.0,
            num_threads=1,
        )
        if not 0 < generated.mesh_stats["tetrahedra"] <= 2_100_000:
            raise ValueError("Mesh exceeds the calibrated tetrahedron budget")

    return replace(
        result,
        mesh=mesh,
        mesh_inputs={"edge_transition_um": 100.0, "mesh_processes": 1},
    )


RECIPE = Recipe(
    name="transmon_readout_coupon_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=geometry,
    settings=SETTINGS,
    description="Full-layout qubit/readout/feedline electrostatic coupon with open feedline reference cuts.",
    reference_ground="chip_ground",
    provenance={
        "recipe_version": 1,
        "mesh_processes": 1,
        "edge_transition_um": 100.0,
        "issue": 933,
        "qpdk_geometry_revision": "5dbc4a0d7ccb50f777191575383f689b53555305",
        "device": "transmon_with_resonator_and_probeline",
        "reference_planes": "Feedline layout ports with open etch caps; readout terminal at assembly o1",
        "metal": "Zero-thickness PEC sheets on silicon",
        "assembly": "Single silicon half-domain",
        "coupling_gap_domain": "14 to 20 um, with a ground strip between the coupled lines; do not interpolate across its disappearance at 12 um",
        "junction": "Open and omitted; junction inductance and capacitance remain explicit",
        "limitations": "Static open-reference coupon. No shorted resonator mode, propagation delay, kinetic inductance or finite-bump field. Requires driven/eigenmode and resonance validation before any parent-circuit replacement.",
        "reduction": "Retain all Maxwell and mutual branches. Do not add full coupon admittance to a distributed resonator model, which already contains route and ground capacitance.",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
