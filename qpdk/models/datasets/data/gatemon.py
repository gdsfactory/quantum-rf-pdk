# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@main",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
r"""Grounded gatemon island with its gate line and a passing readout CPW.

A Palace electrostatic solve gives everything the layout fixes. The device is
about 0.4 mm across, well below a tenth of the ~2.5 mm on-chip wavelength at
5 GHz, so the island couples to its surroundings as a lumped capacitance
network. The Josephson energy is set by the gate voltage through the Andreev
bound states, not by the geometry, so it stays an explicit circuit input
instead of being frozen into an eigenmode or driven solve. The three
terminals give

- ``island``: total capacitance :math:`C_\Sigma`, hence :math:`E_\text{C}`,
- ``gate``: gate capacitance :math:`C_\text{g}`, which sets the lever arm
  :math:`n_\text{g} = C_\text{g} V_\text{g} / 2e` and the qubit decay
  through the gate line,
- ``readout``: coupling capacitance :math:`C_\text{c}` to the resonator.

The readout terminal is a straight CPW segment running over the top of the
pocket, as the coupler section does in ``gatemon_test_chip``. Its ends are cut
open, so its self-capacitance includes the segment's own capacitance to
ground. Do not add that to a distributed resonator model; use only the island
row. The gate line is cut open at the cell's gate port. The junction is open
and omitted: the lower electrode is ground, the upper electrode is island.
Only M1 metal enters the solve, as zero-thickness sheets on silicon.

The junction electrodes stay 15 µm inside the island's left edge and the gate
line at 0.4 × ``island_size`` from the centre, so every grid point is a valid
:func:`~qpdk.cells.gatemon` and the gate stays to the right of the junction.

Preview with ``uv run --script qpdk/models/datasets/data/gatemon.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes
first, with attention to the 1 µm weak-link gap and the gate tip.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
"""

import math
from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import select_metal

from qpdk.cells import gatemon, straight
from qpdk.tech import LAYER

GRID = {
    "island_size": [100.0, 150.0, 200.0, 250.0],
    "pocket_gap": [30.0, 60.0, 90.0],
    "pocket_gap_top": [20.0, 30.0, 40.0, 60.0],
    "gate_tip_gap": [2.0, 5.0, 10.0],
}
SETTINGS = Settings(near_mesh=2.0, far_mesh=80.0, domain_pad=1200.0, order=3)
TERMINALS = ("island", "gate", "readout")
GROUND_MARGIN = 300.0
"""Coplanar ground kept around the device and readout segment, in µm."""
READOUT_OVERHANG = 150.0
"""Length of the readout segment beyond each side wall of the pocket, in µm."""
ELECTRODE_INSET = 15.0
"""Distance from the island's left edge to the junction electrodes, in µm."""
GATE_POSITION = 0.4
"""Gate line centre as a fraction of ``island_size`` right of the island centre."""


def device_settings(point: dict[str, float]) -> dict:
    """Map one grid point to :func:`~qpdk.cells.gatemon` arguments."""
    size = point["island_size"]
    electrode_width = 40.0
    electrode_offset = -(size / 2 - ELECTRODE_INSET - electrode_width / 2)
    return dict(
        island_size=(size, size),
        pocket_gap=point["pocket_gap"],
        pocket_gap_top=point["pocket_gap_top"],
        electrode_size=(electrode_width, 10.0),
        electrode_offset=electrode_offset,
        gate_offset=GATE_POSITION * size - electrode_offset,
        gate_tip_gap=point["gate_tip_gap"],
    )


def _inward(port: gf.Port, distance: float) -> tuple[float, float]:
    """Return the point ``distance`` µm back from ``port`` into its own line."""
    angle = math.radians(port.orientation)
    return (
        port.dx - distance * math.cos(angle),
        port.dy - distance * math.sin(angle),
    )


def _open_end(port: gf.Port, length: float, span: float) -> gf.kdb.Region:
    """Etch an open-circuit gap of ``length`` just beyond a CPW end."""
    cap = gf.Component()
    ref = cap.add_ref(
        gf.components.rectangle(size=(length, span), centered=True, layer=LAYER.M1_ETCH)
    )
    ref.drotate(port.orientation)
    ref.dmove(_inward(port, -length / 2))
    return cap.get_region(LAYER.M1_ETCH)


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Build the gatemon, readout segment and finite coplanar ground."""
    if min(point.values()) <= 0:
        raise ValueError("Island size, gaps and gate tip gap must be positive")
    device = gatemon(**device_settings(point))
    xs = gf.get_cross_section("cpw")
    cpw_gap = max(section.width for section in xs.sections if section.offset)
    span = xs.width + 2 * cpw_gap

    assembly = gf.Component()
    assembly.add_ref(device)
    pocket_half_width = point["island_size"] / 2 + point["pocket_gap"]
    readout = assembly.add_ref(
        straight(length=2 * (pocket_half_width + READOUT_OVERHANG), cross_section=xs)
    )
    # Conductor edge on the top wall of the pocket, as in gatemon_test_chip
    readout.dmove((
        -readout.ports["o1"].dx - pocket_half_width - READOUT_OVERHANG,
        point["island_size"] / 2 + point["pocket_gap_top"] + xs.width / 2,
    ))

    component = gf.Component()
    dbu = component.kcl.dbu
    bounds = assembly.dbbox()
    box = gf.kdb.Box(
        round((bounds.left - GROUND_MARGIN) / dbu),
        round((bounds.bottom - GROUND_MARGIN) / dbu),
        round((bounds.right + GROUND_MARGIN) / dbu),
        round((bounds.top + GROUND_MARGIN) / dbu),
    )
    etch = assembly.get_region(LAYER.M1_ETCH)
    for port in (readout.ports["o1"], readout.ports["o2"], device.ports["gate"]):
        etch |= _open_end(port, cpw_gap, span)
    metal = ((gf.kdb.Region(box) - etch) | assembly.get_region(LAYER.M1_DRAW)).merged()

    gate = device.ports["gate"]
    terminal_ports = {
        "island": gf.Port(
            name="island", center=(0, 0), width=1, orientation=0, layer=LAYER.M1_DRAW
        ),
        # One micrometre back from the open end, on the gate conductor
        "gate": gf.Port(
            name="gate",
            center=_inward(gate, 1.0),
            width=1,
            orientation=gate.orientation,
            layer=LAYER.M1_DRAW,
        ),
        "readout": gf.Port(
            name="readout",
            center=readout.dcenter,
            width=1,
            orientation=90,
            layer=LAYER.M1_DRAW,
        ),
    }
    electrodes = {
        name: select_metal(metal, port, dbu) for name, port in terminal_ports.items()
    }
    ground = metal.interacting(
        gf.kdb.Region(gf.kdb.Box(box.left, box.bottom, box.left + 2, box.bottom + 2))
    )
    names = list(electrodes)
    for i, name in enumerate(names):
        if not (electrodes[name] & ground).is_empty():
            raise ValueError(f"{name} touches chip ground")
        for other in names[:i]:
            if not (electrodes[name] & electrodes[other]).is_empty():
                raise ValueError(f"Electrodes {name} and {other} are shorted")

    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for name, port in terminal_ports.items():
        component.add_port(name, port=port, port_type="electrical")
    pad = settings.domain_pad
    return Geometry(
        component=component,
        bounds=(
            box.left * dbu - pad,
            box.bottom * dbu - pad,
            box.right * dbu + pad,
            box.top * dbu + pad,
        ),
        terminal_ports={name: name for name in TERMINALS},
    )


RECIPE = Recipe(
    name="gatemon_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=geometry,
    settings=SETTINGS,
    description="Grounded QPDK gatemon island, gate line and passing readout CPW in a finite coplanar ground.",
    reference_ground="coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "issue": 1030,
        "metal": "Zero-thickness PEC sheets at the air/silicon interface",
        "ground": f"Coplanar ground {GROUND_MARGIN:g} um beyond the device and readout segment",
        "readout": f"cpw cross-section segment over the pocket top, {READOUT_OVERHANG:g} um beyond each pocket wall, both ends cut open by one CPW gap",
        "gate": "Gate line cut open by one CPW gap at the cell gate port",
        "junction": "Open and omitted; lower electrode grounded, upper electrode part of the island. E_J and any junction capacitance are explicit circuit inputs",
        "layout_rule": f"electrode_offset = -(island_size/2 - {ELECTRODE_INSET:g} - 20) um; gate line at {GATE_POSITION:g} * island_size from the island centre",
        "reduction": "C_sigma = C[island, island] (Maxwell); C_g = -C[island, gate]; C_c = -C[island, readout]. Do not add the readout self-capacitance to a distributed resonator model.",
        "excluded_environment": "No top-gate dielectric or electrode, resonator mode, feedline, package lid or backside ground",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
