# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@610b4d5bf3becdb3f2f78dc695cd9cdc85194e22",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Grounded Xmon island and separate local readout paddle.

All four Xmon arms are one electrical node, not four independent capacitors.
Arm lengths scale together from (160,120,160,120) um and their widths stay
(30,20,30,20) um. The paddle is 20 um wide, beside the left arm. The junction
is open in the solve and connects the island to reference ground in the model.

No distributed CPW enters this coupon. Its full two-node Maxwell matrix can
be used with junction_model(junction=("island", None), exposed_ports=("readout",)).
The inductor and junction capacitance are explicit additional circuit inputs.
The shipped table passes independent h/p/domain checks at five geometries and
held-out checks across the refined grid. Fresh output needs its own validation.
Preview with --dry-run before generating a new grid.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import select_metal

from qpdk.cells import plate_capacitor_single, xmon_transmon
from qpdk.tech import LAYER

GRID = {
    "arm_scale": [0.8, 1.0, 1.2],
    "gap_width": [8.0, 10.0, 12.0],
    "coupling_gap": [3.0, 4.0, 5.0, 7.5, 10.0],
    "paddle_length": [30.0, 37.5, 45.0, 60.0, 75.0, 90.0],
}
SETTINGS = Settings(near_mesh=0.7, far_mesh=48.0, domain_pad=1200.0, order=3)


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Retain the local qubit and readout paddle without a resonator route."""
    if min(point.values()) <= 0:
        raise ValueError("All device dimensions and scales must be positive")
    scale = point["arm_scale"]
    device = xmon_transmon(
        arm_lengths=tuple(scale * value for value in (160.0, 120.0, 160.0, 120.0)),
        gap_width=point["gap_width"],
    )
    names = ("island",)
    ports = (device.ports["left_arm"],)
    local = gf.Component()
    local.add_ref(device)
    paddle = local.add_ref(
        plate_capacitor_single(length=point["paddle_length"], width=20)
    )
    paddle.dmove((
        ports[0].dx - point["coupling_gap"] - 10,
        ports[0].dy,
    ))
    bounds = local.dbbox()
    dbu = local.kcl.dbu
    frame = gf.kdb.Region(
        gf.kdb.DBox(
            bounds.left - 300,
            bounds.bottom - 300,
            bounds.right + 300,
            bounds.top + 300,
        ).to_itype(dbu)
    )
    etch = local.get_region(LAYER.M1_ETCH)
    # The CPW is external to this coupon; keep its port cut electrically open.
    pin = paddle.ports["o1"]
    cut = gf.kdb.DBox(pin.dx - 8, pin.dy - 20, pin.dx, pin.dy + 20)
    etch |= gf.kdb.Region(cut.to_itype(dbu))
    metal = ((frame - etch) | local.get_region(LAYER.M1_DRAW)).merged()
    terminals = (*names, "readout")
    terminal_ports = (*ports, pin)
    electrodes = [select_metal(metal, port, dbu) for port in terminal_ports]
    box = frame.bbox()
    ground = metal.interacting(
        gf.kdb.Region(gf.kdb.Box(box.left, box.bottom, box.left + 2, box.bottom + 2))
    )
    for i, electrode in enumerate(electrodes):
        if not (electrode & ground).is_empty():
            raise ValueError(f"{terminals[i]} touches reference ground")
        if any(not (electrode & other).is_empty() for other in electrodes[:i]):
            raise ValueError("Local qubit electrodes are shorted")
    component = gf.Component()
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for name, port in zip(terminals, terminal_ports, strict=True):
        component.add_port(name, port=port, port_type="electrical")
    pad = settings.domain_pad
    box = frame.bbox().to_dtype(dbu)
    return Geometry(
        component=component,
        bounds=(box.left - pad, box.bottom - pad, box.right + pad, box.top + pad),
        terminal_ports={name: name for name in terminals},
    )


RECIPE = Recipe(
    name="xmon_readout_palace",
    script=Path(__file__),
    grid=GRID,
    axis_units={"arm_scale": "1"},
    terminals=("island", "readout"),
    geometry=geometry,
    settings=SETTINGS,
    description="Local Xmon island and readout paddle, excluding external CPW.",
    reference_ground="finite_coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "metal": "Zero-thickness M1 PEC sheets on silicon",
        "reference_planes": "Paddle CPW pin, open cut at its left edge",
        "junction": "Open and omitted, circuit junction connects island to ground",
        "scope": "Equipotential cross, no distributed arm or resonator modes",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
