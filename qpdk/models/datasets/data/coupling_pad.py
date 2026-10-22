# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@c00764bc18cbcb0e58ccb5c7dfb880d550455d35",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Facing QPDK coupling pads with grounded surroundings, including lateral offset.

At zero offset these are the two halves of plate_capacitor. A single pad is
always simulated with its facing electrode; its self-capacitance is never
substituted for coupling. The finite ground frame is part of this environment,
not an infinite chip ground. All geometry arguments are in micrometres.

Load with capacitance_model(path): o1/o2 are the two metal pads relative to the
frame. This retains both ground branches. A scalar-only circuit may use
-C[0,1] only when its ground branches are supplied separately without overlap.

Preview with ``uv run --script qpdk/models/datasets/data/coupling_pad.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes first.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli

from qpdk.cells import coupling_pad_pair
from qpdk.tech import LAYER

GRID = {
    "length": [40.0, 80.0, 120.0],
    "width": [5.0, 10.0, 20.0],
    "gap": [4.0, 7.0, 10.0],
    "offset": [-12.0, 0.0, 12.0],
}
SETTINGS = Settings(near_mesh=1.0, far_mesh=30.0, domain_pad=200.0, order=3)
TERMINALS = ("o1", "o2")


def geometry(point: dict[str, float], settings: Settings, _processes: int) -> Geometry:
    """Simulate the same paired-pad cell used in layouts."""
    component = gf.Component()
    device = component.add_ref(coupling_pad_pair(**point))
    for name in TERMINALS:
        component.add_port(name, port=device.ports[name], port_type="electrical")
    box = component.dbbox(gf.get_layer(LAYER.M1_DRAW))
    bounds = (
        box.left - settings.domain_pad,
        box.bottom - settings.domain_pad,
        box.right + settings.domain_pad,
        box.top + settings.domain_pad,
    )
    return Geometry(
        component=component,
        bounds=bounds,
        terminal_ports={name: name for name in TERMINALS},
    )


RECIPE = Recipe(
    name="coupling_pad_palace",
    script=Path(__file__),
    grid=GRID,
    settings=SETTINGS,
    terminals=TERMINALS,
    geometry=geometry,
    description="Facing QPDK plate_capacitor_single electrodes with offset and a finite ground frame.",
    reference_ground="coplanar_ground_frame",
    provenance={
        "recipe_version": 1,
        "qpdk_geometry_revision": "5dbc4a0d7ccb50f777191575383f689b53555305",
        "issue": 929,
        "numerical_accuracy": "Nominal p3 refinement from 1/30 um to 0.49/14.7 um changes Maxwell and ground capacitances by less than 1.1%. Representative sensitivity, not a global error bound.",
        "metal": "PEC sheets on a silicon half-domain; no kinetic inductance or junction physics",
        "ground": "Frame 45 um wide, 10 um from the electrode bounding box",
        "terminals": {
            "o1": "first pad",
            "o2": "facing pad shifted vertically by offset",
        },
        "reduction": "Mutual off-diagonal -C[o1,o2] is coupling; mutual diagonal is each pad-ground branch. Full Maxwell SAX admittance includes both.",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
