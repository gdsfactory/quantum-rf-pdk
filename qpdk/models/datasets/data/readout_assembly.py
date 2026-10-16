# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@c00764bc18cbcb0e58ccb5c7dfb880d550455d35",
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

from functools import partial
from pathlib import Path

from _experiment import Recipe, Settings, run_cli
from _readout import TERMINALS, geometry

GRID = {
    "pad_length": [300.0, 394.0],
    "pad_offset": [35.0, 55.0],
    "feedline_gap": [14.0, 15.0, 16.0, 18.0, 20.0],
    "coupling_length": [500.0, 800.0],
}
SETTINGS = Settings(near_mesh=1.4, far_mesh=120.0, domain_pad=2000.0, order=3)

RECIPE = Recipe(
    name="transmon_readout_coupon_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=partial(geometry, flip=False),
    settings=SETTINGS,
    description="Full-layout qubit/readout/feedline electrostatic coupon with open feedline reference cuts.",
    reference_ground="chip_ground",
    provenance={
        "recipe_version": 1,
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
