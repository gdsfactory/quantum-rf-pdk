# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@987b9f6150f4e98ebd59e2778aabcb88144ed7b2",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/gdsfactory/gsim.git@3bab4eb44cf3cf31c3b7fa69b4f0c65c99a91b78",
# ]
# ///
r"""Bare bonded flipmon between two silicon chips.

The 15 um indium bump joins the inner disk and top pad into one terminal.
Both finite chip ground frames share the reference potential. The separate
ring remains a separate terminal. No paddle or distributed resonator is included.

The optional ideal_wire control omits the bump but holds the inner disk and
top pad at the same potential. Comparing it with physical_bump isolates the
bump's electrostatic contribution. Neither variant includes bond inductance.
New solves remain candidates until independently qualified.

The default GRID reproduces all 189 published training coordinates. Preview it
without running Palace from the repository root::

    python qpdk/models/datasets/recipes/flipmon_bonded.py --dry-run

To reproduce qualification from retained native evidence, replace STUDY with
the frozen bare-flipmon-refinement-r35 directory and CALIBRATION with the frozen
bare-flipmon-r31 directory. Keep their source manifests, submission receipts,
summaries, Parquet parts and raw native runs intact. Then run::

    python -m qpdk.models.datasets.recipes.curate_flipmon_bonded \
        STUDY CALIBRATION \
        --report build/flipmon-bonded-verification.json \
        --output build/datasets/flipmon_bonded_palace

Leave out --output to preview qualification. The curator combines 108 new and
81 reused training points and checks 293 native points in total, including
48 fresh cell centres, eight fresh faces, 24 older retuning references and
the numerical and ideal-wire controls. It exports only when every gate passes.
Do not change archived recipes or manifests to match the current default GRID.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
from _experiment import Geometry, Recipe, Settings, run_cli

from qpdk.cells import flipmon_with_bbox
from qpdk.tech import LAYER

if TYPE_CHECKING:
    from gsim.palace import ElectrostaticSim

GRID = {
    "chip_gap": [5.0, 6.25, 7.5, 8.75, 10.0, 12.5, 15.0],
    "alignment": [-15.0, 0.0, 15.0],
    "inner_radius": [50.0, 55.0, 60.0],
    "top_radius": [100.0, 110.0, 120.0],
}
SETTINGS = Settings(near_mesh=1.4, far_mesh=60.0, domain_pad=600.0, order=3)
TERMINALS = ("island", "ring")


def geometry(
    point: dict[str, float | str], settings: Settings, processes: int
) -> Geometry:
    """Select real connected nets, including the finite bump and both metal faces."""
    gap, alignment, inner_radius, top_radius = (float(point[name]) for name in GRID)
    bond = point.get("bond", "physical_bump")
    if gap <= 0 or not 7.5 < inner_radius < 80:
        raise ValueError("Require positive chip gap and 7.5 < inner_radius < 80 um")
    if bond not in {"physical_bump", "ideal_wire"}:
        raise ValueError(f"Unknown bond control {bond!r}")
    if abs(alignment) + 7.5 >= top_radius:
        raise ValueError("The centered bump must land inside the shifted top pad")
    device = flipmon_with_bbox(
        inner_circle_radius=inner_radius, top_circle_radius=top_radius
    )
    dbu = device.kcl.dbu
    frame = gf.kdb.Region(gf.kdb.DBox(-400, -400, 400, 400).to_itype(dbu))
    bottom = (
        (frame - device.get_region(LAYER.M1_ETCH)) | device.get_region(LAYER.M1_DRAW)
    ).merged()
    shifted = gf.Component()
    shifted.add_ref(device.extract([LAYER.M2_DRAW, LAYER.M2_ETCH])).dmovex(alignment)
    top = (
        (frame - shifted.get_region(LAYER.M2_ETCH)) | shifted.get_region(LAYER.M2_DRAW)
    ).merged()
    component = gf.Component()
    regions = [(LAYER.M1_DRAW, bottom), (LAYER.M2_DRAW, top)]
    if bond == "physical_bump":
        regions.append((LAYER.IND, device.get_region(LAYER.IND)))
    for layer, region in regions:
        for polygon in region.each():
            component.add_polygon(polygon.to_dtype(dbu), layer=layer)
    for name, center in (
        ("island", (0, 0)),
        ("ring", (100, 0)),
    ):
        component.add_port(
            name=name,
            center=center,
            width=1,
            orientation=180,
            layer=LAYER.M1_DRAW,
            port_type="electrical",
        )
    top_z = gap + 0.2
    stack = {
        "layers": {
            name: {
                "name": name,
                "gds_layer": gf.get_layer_tuple(layer),
                "zmin": z0,
                "zmax": z1,
                "thickness": z1 - z0,
                "material": "pec",
                "layer_type": kind,
            }
            for name, layer, z0, z1, kind in (
                ("bottom", LAYER.M1_DRAW, 0, 0.2, "conductor"),
                ("top", LAYER.M2_DRAW, top_z, top_z + 0.2, "conductor"),
                ("bump", LAYER.IND, 0.2, top_z, "via"),
            )
        },
        "materials": {
            "pec": {"conductivity": 1e30},
            "silicon": {"permittivity": settings.permittivity},
            "air": {"permittivity": 1.0},
        },
        "dielectrics": [
            {"name": "bottom_chip", "zmin": -300, "zmax": 0, "material": "silicon"},
            {"name": "gap", "zmin": 0, "zmax": top_z + 0.2, "material": "air"},
            {
                "name": "top_chip",
                "zmin": top_z + 0.2,
                "zmax": top_z + 300.2,
                "material": "silicon",
            },
        ],
    }
    airbox = dict(
        margin_x=settings.domain_pad,
        margin_y=settings.domain_pad,
        z_above=settings.domain_pad,
        z_below=settings.domain_pad,
    )
    mesh_options = dict(
        refined_mesh_size=settings.near_mesh,
        max_mesh_size=settings.far_mesh,
        planar_conductors=False,
        merge_via_distance=0,
        decimate_tolerance=0,
        threads=processes,
        surface_threads=1,
        verbose=True,
    )

    def mesh(sim: ElectrostaticSim) -> None:
        """Keep the connected bump on its own island, away from ring and ground."""
        from gsim.common import LayerStack  # ruff: ignore[import-outside-top-level]

        sim.set_stack(LayerStack.model_validate(stack))
        if bond == "physical_bump":
            sim.add_terminal("island", net="island")
        else:
            sim.add_terminal("island", region=(-1, -1, 1, 1))
        for name in TERMINALS[1:]:
            sim.add_terminal(name, net=name)
        resolved = sim.resolve_terminals()
        expected = 1 if bond == "physical_bump" else 2
        if len(resolved[0].nets) != expected or any(
            len(t.nets) != 1 for t in resolved[1:]
        ):
            raise ValueError(
                "Flipmon electrodes are shorted or the bump is disconnected"
            )
        sim.set_airbox(**airbox)
        sim.mesh(**mesh_options)

    pad = 400 + settings.domain_pad
    return Geometry(
        component=component,
        bounds=(-pad, -pad, pad, pad),
        terminal_ports={name: name for name in TERMINALS},
        mesh=mesh,
        mesh_inputs={
            "stack": stack,
            "airbox": airbox,
            "mesh": mesh_options,
            "bond": bond,
        },
    )


RECIPE = Recipe(
    name="flipmon_bonded_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=geometry,
    settings=SETTINGS,
    variants={"bond": ["physical_bump"]},
    description="Bare bonded flipmon with finite-thickness metal and a finite indium bump.",
    reference_ground="both_chip_ground_frames",
    provenance={
        "recipe_version": 1,
        "metal_thickness_um": 0.2,
        "substrate_thickness_um": 300.0,
        "outer_ring": "Radius 110 um and width 60 um",
        "qpdk_geometry_revision": "5fb9e352ff27bb38f773ecb949bbc26a3e525e6b",
        "gsim_geometry_revision": "3bab4eb44cf3cf31c3b7fa69b4f0c65c99a91b78",
        "reference_planes": "Two equipotential electrodes, with no distributed ports or readout paddle",
        "junction": "Open and omitted, circuit junction connects island and ring",
        "bond": "physical_bump includes a 15 um PEC bump; ideal_wire omits its geometry and imposes equal island and top-pad potential",
        "ground": "Two 800 um square chip frames at the same reference potential",
        "limitations": "Candidate bare electrostatic block. No numerical admission, readout paddle, distributed line, kinetic inductance, bump inductance or nonlinear spectrum.",
        "admission_status": "candidate_pending_independent_numerical_validation",
    },
)

if __name__ == "__main__":
    run_cli(RECIPE)
