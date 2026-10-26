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
Mesh settings are provisional until independent h/p/domain and held-out
references pass. Preview with --dry-run before generating a new grid.
"""

import hashlib
from functools import partial
from pathlib import Path

from _experiment import Recipe, Settings, run_cli
from _planar_qubit import geometry

GRID = {
    "pad_width": [200.0, 250.0, 300.0],
    "ground_clearance": [100.0, 200.0, 300.0],
    "coupling_gap": [5.0, 10.0, 20.0],
    "paddle_length": [100.0, 200.0, 300.0],
}
SETTINGS = Settings(near_mesh=1.5, far_mesh=75.0, domain_pad=1600.0, order=3)
RECIPE = Recipe(
    name="double_pad_readout_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=("left_pad", "right_pad", "readout"),
    geometry=partial(geometry, xmon=False),
    settings=SETTINGS,
    description="Local two-island transmon and readout paddle, excluding external CPW.",
    reference_ground="finite_coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "metal": "Zero-thickness M1 PEC sheets on silicon",
        "reference_planes": "Paddle CPW pin, open cut at its left edge",
        "junction": "Open and omitted, L_J and additional C_J remain circuit inputs",
        "scope": "Local capacitance only, no resonator, propagation or package",
        "geometry_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("_planar_qubit.py").read_bytes()
        ).hexdigest(),
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
