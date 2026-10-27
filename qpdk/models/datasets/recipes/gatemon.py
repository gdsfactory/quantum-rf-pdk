# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@1c71d23bd1682792268a83eda0ad16e965e55878",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
r"""Grounded gatemon island with its gate line and a passing readout CPW.

A Palace electrostatic solve gives everything the layout fixes. The device is
about 0.4 mm across, well below a tenth of the ~25 mm on-chip wavelength at
5 GHz, so the island couples to its surroundings as a lumped capacitance
network. The Josephson energy is set by the gate voltage through the Andreev
bound states, not by the geometry, so it stays an explicit circuit input
instead of being frozen into an eigenmode or driven solve. The three
terminals give

- ``island``: total capacitance :math:`C_\Sigma`, hence :math:`E_\text{C}`,
- ``gate``: gate capacitance :math:`C_\text{g}`, which sets the island offset charge
  :math:`n_\text{g} = C_\text{g} V_\text{g} / 2e` and the qubit decay
  through the gate line,
- ``readout``: coupling capacitance :math:`C_\text{c}` to the resonator.

The readout terminal is a straight CPW segment running over the top of the
pocket, as the coupler section does in ``gatemon_test_chip``. Its ends are cut
open, so its self-capacitance includes the segment's own capacitance to
ground. Do not add that to a distributed resonator model; use only the island
row. The gate line is cut open at the cell's gate port. The junction is open
and omitted: the lower electrode is ground, the upper electrode is island.
Only M1 metal enters the solve, as zero-thickness sheets on silicon. This is
a coarse estimate of the island's remote capacitances. The Nb/Au contact
leads, Al top gate, hBN and thermal oxide are omitted, so this recipe cannot
predict the nanotube's gate lever arm or local junction capacitance. A full
stack calculation needs a mesh resolving the transferred flake and gate.

The nominal point reproduces the default :func:`~qpdk.cells.gatemon` geometry.
Below a 150 µm island size, contact, flake and gate-tip dimensions scale with
the island. The gate feed stays at least 0.4 × ``island_size`` from the centre
and moves right when needed to preserve its hBN landing clearance.

Preview with ``uv run --script qpdk/models/datasets/recipes/gatemon.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes
first, with attention to the CPW slots and gate landing.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
"""

import math
from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import select_metal

from qpdk.cells import gatemon, gatemon_coupler
from qpdk.tech import LAYER

GRID = {
    "island_size": [100.0, 150.0, 200.0, 250.0],
    "pocket_gap": [30.0, 60.0, 90.0],
    "pocket_gap_top": [20.0, 30.0, 40.0, 60.0],
    "gate_tip_gap": [2.0, 5.0, 10.0],
}
SETTINGS = Settings(near_mesh=0.8, far_mesh=60.0, domain_pad=1200.0, order=3)
TERMINALS = ("island", "gate", "readout")
GROUND_MARGIN = 300.0
"""Coplanar ground kept around the device and readout segment, in µm."""
ELECTRODE_INSET = 5.0
"""Distance from the island's left edge to the junction electrodes, in µm."""
GATE_POSITION = 0.4
"""Gate line centre as a fraction of ``island_size`` right of the island centre."""


def device_settings(point: dict[str, float]) -> dict:
    """Map one grid point to :func:`~qpdk.cells.gatemon` arguments."""
    size = point["island_size"]
    scale = min(size / 150.0, 1.0)
    electrode_width = 60.0 * scale
    electrode_offset = -(size / 2 - ELECTRODE_INSET * scale - electrode_width / 2)
    hbn_width = 80.0 * scale
    junction_offset = 20.0 * scale
    tip_width = 16.0 * scale
    hbn_left = electrode_offset + (1.0 + junction_offset - tip_width / 2) / 2
    gate_x = max(
        GATE_POSITION * size,
        hbn_left
        + hbn_width
        + point["gate_tip_gap"]
        + gf.get_cross_section("cpw").width / 2
        + 3.5 * scale,
    )
    return dict(
        island_size=(size, size),
        pocket_gap=point["pocket_gap"],
        pocket_gap_top=point["pocket_gap_top"],
        electrode_size=(electrode_width, electrode_width),
        size_patch=(electrode_width, 50.0 * scale),
        hbn_size=(hbn_width, 80.0 * scale),
        junction_offset=junction_offset,
        gate_tip_width=tip_width,
        electrode_offset=electrode_offset,
        gate_offset=gate_x - electrode_offset,
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
    readout = assembly.add_ref(
        gatemon_coupler(
            island_width=point["island_size"],
            pocket_gap_top=point["pocket_gap_top"],
        )
    )
    readout.connect("qubit", device.ports["readout"])

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
        "recipe_version": 3,
        "issue": 1030,
        "metal": "Zero-thickness PEC sheets at the air/silicon interface",
        "ground": f"Coplanar ground {GROUND_MARGIN:g} um beyond the device and readout segment",
        "readout": "Default gatemon_coupler CPW section attached at its qubit reference plane, matching gatemon_test_chip. Both ends are cut open by one CPW gap",
        "gate": "Gate line cut open by one CPW gap at the cell gate port",
        "junction": "Open and omitted; lower electrode grounded, upper electrode part of the island. E_J and any junction capacitance are explicit circuit inputs",
        "layout_rule": f"Default gatemon dimensions at island_size=150 um; contact, flake and tip dimensions scale by min(island_size/150, 1). Electrodes are inset {ELECTRODE_INSET:g} um times this scale. Gate feed is at least {GATE_POSITION:g} * island_size from the island centre, shifted right as needed for hBN clearance and a 3.5 um times scale landing margin",
        "reduction": "C_sigma = C[island, island] (Maxwell); C_g = -C[island, gate]; C_c = -C[island, readout]. Do not add the readout self-capacitance to a distributed resonator model.",
        "excluded_environment": "M1-only approximation. No Nb/Au contact leads, top-gate dielectric or electrode, thermal oxide, resonator mode, feedline, package lid or backside ground. Does not determine the nanotube gate lever arm",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
