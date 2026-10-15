# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@5dbc4a0d7ccb50f777191575383f689b53555305",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Finite-thickness flipmon electrodes between two silicon chips.

The three-terminal extraction is an unbonded assembly coupon. The indium bump
is omitted so each electrode's capacitance remains separately available. An
ideal electrical bond can be applied afterward with B.T @ C @ B, where B maps
inner_pad and top_pad to the same potential. This omits the bump's fringing
field and needs comparison with a bonded mesh before device use.

The bottom metal occupies z=[0,0.2], the top metal z=[gap+0.2,gap+0.4]. Both
silicon chips are 300 um thick. Opposing metal faces are separated by gap.
Alignment moves only the top chip's circular pad; the finite ground frames
remain fixed. The Josephson junction is open and omitted.

Preview with ``uv run --script qpdk/models/datasets/data/flipmon.py --dry-run``.
Edit GRID and SETTINGS to define a new sweep; validate representative meshes first.
Generate disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
An ideal-bond circuit uses ``junction_model(path, junction=("inner_pad",
"outer_ring"), inductance=L_J, terminal_groups=(("inner_pad", "top_pad"),
("outer_ring",)))``. This ideal reduction does not validate a finite bump.
"""

from pathlib import Path

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli, stack_geometry

from qpdk.cells import flipmon_with_bbox
from qpdk.tech import LAYER

GRID = {
    "chip_gap": [5.0, 7.5, 10.0, 15.0, 20.0],
    "alignment": [-15.0, 15.0],
    "inner_radius": [50.0, 60.0],
    "top_radius": [100.0, 120.0],
}
SETTINGS = Settings(near_mesh=2.8, far_mesh=70.0, domain_pad=400.0)
TERMINALS = ("inner_pad", "outer_ring", "top_pad")


def geometry(point: dict[str, float], settings: Settings, processes: int) -> Geometry:
    """Use public gsim stack meshing, with one extraction layer per electrode."""
    if point["chip_gap"] <= 0 or not 0 < point["inner_radius"] < 80:
        raise ValueError(
            "Require positive chip gap and inner_radius below the ring's 80 um inner edge"
        )
    device = flipmon_with_bbox(
        inner_circle_radius=point["inner_radius"], top_circle_radius=point["top_radius"]
    )
    bottom = device.get_region(LAYER.M1_DRAW).merged()
    polygons = sorted(bottom.each(), key=lambda polygon: polygon.area())
    if len(polygons) != 2:
        raise ValueError(
            "Expected a separate inner disk and ring, with no junction metal"
        )
    top_z = point["chip_gap"] + 0.2
    top = gf.Component()
    top.add_ref(device.extract([LAYER.M2_DRAW])).dmovex(point["alignment"])
    electrodes = {
        "inner_pad": (gf.kdb.Region(polygons[0]), 0.0, 0.2),
        "outer_ring": (gf.kdb.Region(polygons[1]), 0.0, 0.2),
        "top_pad": (top.get_region(LAYER.M2_DRAW), top_z, top_z + 0.2),
    }
    for name, z in [("bottom_ground", 0.0), ("top_ground", top_z)]:
        outer = gf.components.rectangle(
            size=(800, 800), centered=True, layer=LAYER.M1_DRAW
        )
        inner = gf.components.circle(
            radius=170, layer=LAYER.M1_DRAW, angle_resolution=5
        )
        frame = gf.boolean(outer, inner, operation="not", layer=LAYER.M1_DRAW)
        electrodes[name] = (frame.get_region(LAYER.M1_DRAW), z, z + 0.2)
    return stack_geometry(
        electrodes,
        terminals=TERMINALS,
        settings=settings,
        processes=processes,
        dielectrics=[
            {"name": "bottom_chip", "zmin": -300.0, "zmax": 0.0, "material": "silicon"},
            {"name": "gap", "zmin": 0.0, "zmax": top_z + 0.2, "material": "air"},
            {
                "name": "top_chip",
                "zmin": top_z + 0.2,
                "zmax": top_z + 300.2,
                "material": "silicon",
            },
        ],
    )


RECIPE = Recipe(
    name="flipmon_unbonded_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    settings=SETTINGS,
    geometry=geometry,
    description="Unbonded three-electrode QPDK flipmon coupon between two finite silicon chips.",
    reference_ground="both_chip_ground_frames",
    provenance={
        "recipe_version": 1,
        "issue": 931,
        "numerical_accuracy": "Nominal p2 at 2.8/70 um agrees with the 4/100 um p3 comparison within 1.5% for Maxwell and ground branches. The coarse p2 doubled-domain check changes less than 0.1%. Representative sensitivity, not a global error bound.",
        "metal_thickness_um": 0.2,
        "substrate_thickness_um": 300.0,
        "outer_ring": "Radius 110 um, width 60 um; inner pad and top pad from flipmon_with_bbox",
        "ground": "Two 800 um square frames with radius 170 um circular openings, connected to reference ground",
        "junction": "Open and omitted; Josephson inductance and junction capacitance supplied separately",
        "bond": "Unbonded Maxwell matrix. Ideal inner/top bond may be reduced by B.T C B; finite bump fringing is not included or validated.",
        "limitations": "Preparation coupon, not yet a validated bonded flipmon dataset. Ring and pad polygon tessellation follows the QPDK cell.",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
