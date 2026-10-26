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
Mesh settings are provisional until independent numerical controls pass.
Preview with --dry-run before generating a new grid.
"""

import hashlib
from functools import partial
from pathlib import Path

from _experiment import Recipe, Settings, run_cli
from _planar_qubit import geometry

GRID = {
    "arm_scale": [0.8, 1.0, 1.2],
    "gap_width": [8.0, 10.0, 12.0],
    "coupling_gap": [3.0, 5.0, 10.0],
    "paddle_length": [30.0, 60.0, 90.0],
}
SETTINGS = Settings(near_mesh=1.0, far_mesh=60.0, domain_pad=1200.0, order=3)
RECIPE = Recipe(
    name="xmon_readout_palace",
    script=Path(__file__),
    grid=GRID,
    axis_units={"arm_scale": "1"},
    terminals=("island", "readout"),
    geometry=partial(geometry, xmon=True),
    settings=SETTINGS,
    description="Local Xmon island and readout paddle, excluding external CPW.",
    reference_ground="finite_coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "metal": "Zero-thickness M1 PEC sheets on silicon",
        "reference_planes": "Paddle CPW pin, open cut at its left edge",
        "junction": "Open and omitted, circuit junction connects island to ground",
        "scope": "Equipotential cross, no distributed arm or resonator modes",
        "geometry_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("_planar_qubit.py").read_bytes()
        ).hexdigest(),
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
