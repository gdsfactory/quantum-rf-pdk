# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@c00764bc18cbcb0e58ccb5c7dfb880d550455d35",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Two floating transmon islands inside the boxed device's coplanar ground.

Only M1 pad metal enters the solve. The junction is omitted and open; its
capacitance and Josephson inductance remain explicit parallel circuit inputs.
No coupling pad, resonator or other neighbouring device is included.

Use differential_capacitance(lookup(...)) for the equal/opposite charge mode,
then E_C/h = e**2/(2*h*(C_diff+C_junction)). This allows the common mode to float.
The two island SAX ports reference chip ground, not each other.

Preview with ``uv run --script qpdk/models/datasets/data/double_pad_transmon.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes first.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
For a linearized circuit, use ``junction_model(path, junction=("left_pad",
"right_pad"), inductance=L_J, junction_capacitance=C_J)``. L_J is explicit, in H.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, ground_frame, run_cli

from qpdk.cells import double_pad_transmon_with_bbox
from qpdk.tech import LAYER

GRID = {
    "pad_width": [150.0, 250.0, 350.0],
    "pad_height": [250.0, 400.0, 550.0],
    "pad_gap": [10.0, 15.0, 20.0, 30.0, 40.0],
    "ground_clearance": [100.0, 200.0, 300.0],
}
SETTINGS = Settings(near_mesh=2.0, far_mesh=75.6, domain_pad=1600.0, order=4)
TERMINALS = ("left_pad", "right_pad")


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Use the boxed cell's electrode geometry with a finite exterior ground."""
    if min(point.values()) <= 0:
        raise ValueError("Pad dimensions, gap and ground clearance must be positive")
    device = double_pad_transmon_with_bbox(
        pad_size=(point["pad_width"], point["pad_height"]),
        pad_gap=point["pad_gap"],
        bbox_extension=point["ground_clearance"],
    )
    component = gf.Component()
    component.add_ref(device.extract([LAYER.M1_DRAW]))
    for name in TERMINALS:
        component.add_port(name, port=device.ports[name], port_type="electrical")
    bounds = ground_frame(
        component,
        clearance=point["ground_clearance"],
        width=300.0,
        domain_pad=settings.domain_pad,
    )
    return Geometry(
        component=component,
        bounds=bounds,
        terminal_ports={name: name for name in TERMINALS},
    )


RECIPE = Recipe(
    name="double_pad_transmon_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=geometry,
    settings=SETTINGS,
    description="Two floating QPDK transmon M1 islands with boxed clearance and a finite chip ground frame.",
    reference_ground="coplanar_chip_ground_frame",
    provenance={
        "recipe_version": 1,
        "qpdk_geometry_revision": "5dbc4a0d7ccb50f777191575383f689b53555305",
        "issue": 930,
        "metal": "Zero-thickness PEC sheets at the air/silicon interface",
        "ground": "300 um wide frame outside the boxed cell clearance; finite chip environment",
        "junction": "JJ layers omitted, open junction region; C_junction and L_J are separate explicit inputs",
        "reduction": "C_diff = 1/(d.T @ inverse(C) @ d), d=(1,-1). E_C=e^2/(2*(C_diff+C_junction)). Both islands have opposite incremental charge and free common-mode potential.",
        "excluded_environment": "No readout pad, resonator, feedline, package lid or backside ground",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
