# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@5dbc4a0d7ccb50f777191575383f689b53555305",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Equipotential meander-to-ground capacitance, with discrete turn counts.

The entire connected winding is one electrostatic terminal. This measures its
common-mode capacitance to the specified return frame, not inter-turn
capacitance under a varying voltage or a self-resonance model. The two ends
must never be assigned separate electrostatic terminal potentials.

gsim at the pinned revision has MagnetostaticConfig and static solver metrics,
but no MagnetostaticSim, current-loop mesher or inductance-matrix loader.
Geometric inductance extraction therefore remains an upstream prerequisite.
No geometric or total superconducting inductance is fabricated by this script.
Sheet kinetic inductance remains an independently specified model input.

Preview with ``uv run --script qpdk/models/datasets/data/meander_inductor.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes first.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, ground_frame, run_cli

from qpdk.cells import meander_inductor
from qpdk.tech import coplanar_waveguide

GRID = {
    "turn_length": [100.0, 200.0, 300.0],
    "wire_width": [1.0, 2.0],
    "wire_gap": [2.0, 4.0, 8.0],
}
SETTINGS = Settings(near_mesh=0.7, far_mesh=40.0, domain_pad=300.0)
TERMINALS = ("winding",)


def geometry(
    point: dict[str, float | str], settings: Settings, _processes: int
) -> Geometry:
    """Keep the winding continuous and give it one voltage terminal."""
    if min(float(point[name]) for name in GRID) <= 0:
        raise ValueError("Winding dimensions must be positive")
    device = meander_inductor(
        n_turns=int(point["n_turns"]),
        turn_length=float(point["turn_length"]),
        wire_gap=float(point["wire_gap"]),
        cross_section=coplanar_waveguide(width=float(point["wire_width"]), gap=2.0),
    )
    component = gf.Component()
    component.add_ref(device)
    component.add_port("winding", port=device.ports["o1"], port_type="electrical")
    bounds = ground_frame(
        component, clearance=20.0, width=100.0, domain_pad=settings.domain_pad
    )
    return Geometry(
        component=component, bounds=bounds, terminal_ports={"winding": "winding"}
    )


RECIPE = Recipe(
    name="meander_ground_capacitance_palace",
    script=Path(__file__),
    grid=GRID,
    variants={"n_turns": ["3", "5", "7"]},
    terminals=TERMINALS,
    settings=SETTINGS,
    geometry=geometry,
    description="Equipotential winding-to-ground capacitance of QPDK meanders; not a self-resonance extraction.",
    reference_ground="coplanar_return_frame",
    provenance={
        "recipe_version": 1,
        "issue": 932,
        "metal": "Zero-thickness PEC winding at the air/silicon interface",
        "return_geometry": "100 um wide coplanar frame, 20 um from the winding bounding box",
        "electrostatic_terminal": "o1 and o2 are one connected winding; one equipotential terminal",
        "magnetostatic_status": "Not available through the pinned gsim simulation/mesh/result API; requires upstream support and an explicit closed excitation/return loop",
        "kinetic_inductance": "Not extracted; specify sheet kinetic inductance independently",
        "limitations": "No differential parasitic capacitance, geometric inductance or validated self-resonance frequency is inferred from this common-mode solve",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
