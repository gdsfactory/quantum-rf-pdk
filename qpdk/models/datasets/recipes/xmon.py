# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@987b9f61",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Bare equipotential Xmon island inside a finite coplanar ground frame.

All four arms share one electrical node. Arm lengths scale together from
(160,120,160,120) um and widths remain (30,20,30,20) um. This single-terminal
coupon omits junction metal, readout paddles, routes and neighbouring devices.
Its capacitance cannot replace the full Xmon-and-paddle Maxwell matrix.

The junction is open during extraction. Additional junction capacitance and
Josephson inductance are separate circuit inputs. Fresh output remains
provisional until independent mesh, order, domain and held-out checks pass.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli
from _readout import select_metal

from qpdk.cells import xmon_transmon
from qpdk.tech import LAYER

GRID = {
    "arm_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
    "gap_width": [8.0, 9.0, 10.0, 11.0, 12.0],
}
SETTINGS = Settings(near_mesh=0.7, far_mesh=48.0, domain_pad=1200.0, order=3)


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Retain the cross as one conductor and omit all non-M1 junction layers."""
    if min(point.values()) <= 0:
        raise ValueError("Arm scale and etched gap must be positive")
    device = xmon_transmon(
        arm_lengths=tuple(point["arm_scale"] * value for value in (160, 120, 160, 120)),
        gap_width=point["gap_width"],
    )
    dbu = device.kcl.dbu
    etch = device.get_region(LAYER.M1_ETCH)
    box = etch.bbox().to_dtype(dbu)
    frame = gf.kdb.Region(
        gf.kdb.DBox(
            box.left - 300, box.bottom - 300, box.right + 300, box.top + 300
        ).to_itype(dbu)
    )
    metal = ((frame - etch) | device.get_region(LAYER.M1_DRAW)).merged()
    pin = device.ports["left_arm"]
    island = select_metal(metal, pin, dbu)
    for name in ("top_arm", "right_arm", "bottom_arm"):
        if not (select_metal(metal, device.ports[name], dbu) ^ island).is_empty():
            raise ValueError("The Xmon arms must share one electrical node")
    box = frame.bbox()
    ground = metal.interacting(
        gf.kdb.Region(gf.kdb.Box(box.left, box.bottom, box.left + 2, box.bottom + 2))
    )
    if ground.is_empty() or not (island & ground).is_empty() or metal.count() != 2:
        raise ValueError("The bare island must be separate from its ground frame")
    component = gf.Component()
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    component.add_port("island", port=pin, port_type="electrical")
    pad = settings.domain_pad
    box = box.to_dtype(dbu)
    return Geometry(
        component=component,
        bounds=(box.left - pad, box.bottom - pad, box.right + pad, box.top + pad),
        terminal_ports={"island": "island"},
    )


RECIPE = Recipe(
    name="xmon_palace",
    script=Path(__file__),
    grid=GRID,
    axis_units={"arm_scale": "1"},
    terminals=("island",),
    geometry=geometry,
    settings=SETTINGS,
    description="Bare QPDK Xmon cross inside a finite coplanar chip ground frame.",
    reference_ground="finite_coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "metal": "Zero-thickness M1 PEC sheets on silicon",
        "ground": "300 um wide frame outside the etched cross moat",
        "junction": "Open and omitted, circuit junction connects island to ground",
        "scope": "Bare equipotential cross, no readout paddle, distributed arms, routes or package",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
