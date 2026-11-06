# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@610b4d5bf3becdb3f2f78dc695cd9cdc85194e22",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Floating double-pad qubit with a separate local readout paddle.

The M1 sheets, etched pocket and paddle follow the layout cells. Pad height
is 400 um, pad gap 15 um, and paddle width 20 um. No distributed readout
line enters the coupon. All capacitances, including paddle-to-ground, belong
inside this local boundary. Connect the external CPW at the paddle's port.

The junction is open and omitted. Use junction_model with explicit Josephson
inductance and additional junction capacitance. With exposed_ports=("readout",)
the two islands carry no external current. They must not have matched loads.
The candidate mesh passes independent h/p/domain checks at five geometries.
The refined grid still requires held-out interpolation checks before publication.
Preview with --dry-run before generating a new grid.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import select_metal

from qpdk.cells import double_pad_transmon_with_bbox, plate_capacitor_single
from qpdk.tech import LAYER

GRID = {
    "pad_width": [200.0, 250.0, 300.0],
    "ground_clearance": [100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0],
    "coupling_gap": [5.0, 7.5, 10.0, 15.0, 20.0, 27.5, 35.0],
    "paddle_length": [100.0, 150.0, 200.0, 297.0, 394.0],
}
SETTINGS = Settings(near_mesh=1.05, far_mesh=60.0, domain_pad=1600.0, order=3)


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Retain the local qubit and readout paddle without a resonator route."""
    if min(point.values()) <= 0:
        raise ValueError("All device dimensions and scales must be positive")
    device = double_pad_transmon_with_bbox(
        pad_size=(point["pad_width"], 400.0),
        pad_gap=15.0,
        bbox_extension=point["ground_clearance"],
    )
    names = ("left_pad", "right_pad")
    ports = (device.ports["left_pad"], device.ports["right_pad"])
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
    name="double_pad_readout_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=("left_pad", "right_pad", "readout"),
    geometry=geometry,
    settings=SETTINGS,
    description="Local two-island transmon and readout paddle, excluding external CPW.",
    reference_ground="finite_coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "metal": "Zero-thickness M1 PEC sheets on silicon",
        "reference_planes": "Paddle CPW pin, open cut at its left edge",
        "junction": "Open and omitted, L_J and additional C_J remain circuit inputs",
        "scope": "Local capacitance only, no resonator, propagation or package",
        "nominal_layout": {
            "pad_width": 250.0,
            "ground_clearance": 200.0,
            "coupling_gap": 35.0,
            "paddle_length": 394.0,
        },
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
