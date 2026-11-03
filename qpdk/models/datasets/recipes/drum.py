# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@df7a82183721f645033c71bfbde47e61b8af0c4e",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
r"""Coupling and parasitic capacitance of the drum LC resonator.

A Palace electrostatic solve of the full default
:func:`~qpdk.cells.drum_lc_resonator` layout, with its feedline, in a finite
coplanar ground. It supplies the two capacitances the lumped model
:func:`qpdk.models.drum.drum_lc_resonator` takes as inputs: the coupling
:math:`C_\text{c}` to the feedline and the parasitic :math:`C_\text{p}` in
parallel with the drum. It also checks the drum capacitance :math:`C_\text{d}`.

Terminals
    ``node`` is the coupling pad, its wiring and the drum bottom electrode.
    ``membrane`` is the ``VG_TOP`` membrane, a separate conductor above the
    vacuum gap. ``feed`` is the feedline centre conductor, cut open one CPW gap
    beyond both cell ports. ``meander_1`` to ``meander_K`` are blocks of
    consecutive meander runs, numbered from the grounded end. One-micrometre
    slits separate the meander from ground, from the node and between blocks.
    The drum ring and the rest of the metal are ground.

Reduction
    At the lumped resonance the inductor current is uniform, so the meander
    voltage rises linearly from ground to the node. Block :math:`k` gets the
    weight :math:`v_k`, the mean position of its runs along the wire, the node
    gets 1, and the feed and membrane 0. With :math:`C` the Maxwell matrix,
    :math:`C_\text{c} = -\sum_j v_j C_{\text{feed},j}`,
    :math:`C_\text{d} = -\sum_j v_j C_{\text{membrane},j}` and
    :math:`C_\text{p} = v^\mathsf{T} C v - C_\text{c} - C_\text{d}`. The weights
    are stored in the provenance; the full matrix is kept for other weightings.
    The default meander is about 23 mm of wire, longer than the guided
    wavelength at 7 GHz, so a linear voltage profile is itself an
    approximation. ``meander_inductor.py`` is the driven-solve route for the
    distributed answer.

Vacuum gap
    The sacrificial layer is released, so the gap is vacuum. The membrane is a
    disc at the ``VacuumGap_Top`` height without its landing posts. That is
    exact for every node quantity: the Maxwell column of an excited node has all
    other conductors at zero, as the real membrane is through its posts and the
    ring. Only ``C[membrane, membrane]`` misses the posts. The suspended part
    of the membrane also covers the electrode lead in the ring slot, so the
    extracted :math:`C_\text{d}` includes that strip, which the analytical
    model leaves out.

Sheets or solids
    ``CONDUCTORS = "sheet"`` (default) draws every conductor as a
    zero-thickness PEC sheet, M1 on the silicon surface and the membrane one
    vacuum gap above it, as the other QPDK recipes treat M1. ``"solid"``
    extrudes M1 and the membrane to their stack thicknesses with the gap
    between the top of M1 and the membrane. Run the convergence study for both
    before choosing. A 2D sheet solve without the membrane cannot hold
    :math:`C_\text{d}` and would double count the electrode's field into the air
    half-space, so it is not offered.

Mesh
    ``near_mesh`` refines along every conductor edge. A gmsh box field forces
    ``DRUM_MESH`` inside the vacuum gap under the membrane, growing to the far
    size over ``DRUM_MESH_TRANSITION``. The gap field is nearly uniform, so one
    layer of cubic elements across it should suffice; the convergence study
    checks half and twice that size.

Preview with ``uv run --script qpdk/models/datasets/recipes/drum.py --dry-run``.
``uv run --script qpdk/models/datasets/recipes/drum.py convergence --dry-run``
lists the mesh study at the nominal point. Update the ``qpdk`` pin in the
script header to a commit that contains this recipe before running. Generate
disjoint grids with ``--shard``/``--shards`` and merge with repeated
``--merge-shards`` arguments only after every shard succeeds.
"""

import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal

import gdsfactory as gf
import typer
from _experiment import (
    Geometry,
    Recipe,
    Settings,
    generate,
    run_cli,
    stack_geometry,
)

from qpdk import PDK, logger
from qpdk.cells import drum_lc_resonator, meander_inductor
from qpdk.models.drum import DRUM_INDUCTOR_TURNS
from qpdk.tech import LAYER, LAYER_STACK

GRID = {
    "coupling_gap": [3.0, 6.0, 12.0],
    "coupling_pad_size": [10.0, 20.0, 40.0],
    "radius": [5.0, 7.5, 10.0],
    "turn_length": [200.0, 300.0],
}
NOMINAL = {
    "coupling_gap": 6.0,
    "coupling_pad_size": 20.0,
    "radius": 7.5,
    "turn_length": 300.0,
}
"""Default :func:`~qpdk.cells.drum_lc_resonator` geometry."""
SETTINGS = Settings(near_mesh=1.0, far_mesh=60.0, domain_pad=600.0, order=3)
N_TURNS = DRUM_INDUCTOR_TURNS
SEGMENTS = 5
"""Meander blocks; the linear-profile weights then miss the continuum by ~1%."""
TERMINALS = (
    "node",
    "membrane",
    "feed",
    *(f"meander_{k + 1}" for k in range(SEGMENTS)),
)
GROUND_MARGIN = 300.0
"""Coplanar ground kept around the cell, in µm."""
CUT = 1.0
"""Width of the etched slits that split the meander into terminals, in µm."""
CONDUCTORS: Literal["sheet", "solid"] = "sheet"
DRUM_MESH = 0.07
"""Element size inside the vacuum gap, in µm (one element across the 70 nm gap)."""
DRUM_MESH_TRANSITION = 2.0
"""Distance over which the gap size relaxes to the far size, in µm."""


def segment_runs(n_turns: int = N_TURNS, segments: int = SEGMENTS) -> list[int]:
    """Return the first run of every block and ``n_turns`` at the end."""
    if not 1 <= segments <= n_turns:
        raise ValueError("Require 1 <= segments <= n_turns")
    return [round(k * n_turns / segments) for k in range(segments + 1)]


def segment_weights(n_turns: int = N_TURNS, segments: int = SEGMENTS) -> list[float]:
    """Linear-profile voltage of each block, relative to the node."""
    edges = segment_runs(n_turns, segments)
    return [(a + b) / (2 * n_turns) for a, b in pairwise(edges)]


def terminal_weights(n_turns: int = N_TURNS, segments: int = SEGMENTS) -> list[float]:
    """Voltage of each terminal at the lumped resonance, in ``TERMINALS`` order."""
    return [1.0, 0.0, 0.0, *segment_weights(n_turns, segments)]


def _box(x0: float, y0: float, x1: float, y1: float, dbu: float) -> gf.kdb.Region:
    """Return a box region from µm coordinates."""
    return gf.kdb.Region(
        gf.kdb.Box(
            round(min(x0, x1) / dbu),
            round(min(y0, y1) / dbu),
            round(max(x0, x1) / dbu),
            round(max(y0, y1) / dbu),
        )
    )


def _pick(
    metal: gf.kdb.Region, x: float, y: float, dbu: float, name: str
) -> gf.kdb.Region:
    """Select the single polygon under a point."""
    probe = gf.kdb.Region(
        gf.kdb.Box(
            round(x / dbu) - 2,
            round(y / dbu) - 2,
            round(x / dbu) + 2,
            round(y / dbu) + 2,
        )
    )
    selected = metal.interacting(probe)
    if selected.count() != 1:
        raise ValueError(
            f"{name} probe at ({x:g}, {y:g}) selects {selected.count()} polygons"
        )
    return selected


def layout(point: dict[str, float]) -> dict[str, Any]:
    """Build the cell and the solver regions, and check the nets.

    Returns:
        ``electrodes`` (name to region, terminals then ``ground``), the
        ``membrane`` bounding box and the slit regions ``cuts``.

    Raises:
        ValueError: If a dimension is not positive or the nets are wrong.
    """
    if min(point.values()) <= 0:
        raise ValueError("Gaps, sizes, radius and turn length must be positive")
    PDK.activate()
    cell = drum_lc_resonator(
        radius=point["radius"],
        n_turns=N_TURNS,
        turn_length=point["turn_length"],
        coupling_gap=point["coupling_gap"],
        coupling_pad_size=point["coupling_pad_size"],
    )
    dbu = cell.kcl.dbu
    feed_xs = gf.get_cross_section("cpw")
    feed_gap = max(section.width for section in feed_xs.sections if section.offset)
    wire_xs = gf.get_cross_section("meander_inductor_cross_section")
    w = wire_xs.width

    # Rebuild the meander placement of drum_lc_resonator
    meander = meander_inductor(
        n_turns=N_TURNS,
        turn_length=point["turn_length"],
        cross_section="meander_inductor_cross_section",
        add_etch=False,
    )
    start, end = meander.ports["o1"].dcenter, meander.ports["o2"].dcenter
    pitch = (end[1] - start[1]) / (N_TURNS - 1)
    pad_bottom = -feed_xs.width / 2 - point["coupling_gap"] - point["coupling_pad_size"]
    node_y = pad_bottom - 10.0 - w / 2  # cell default margin
    shift = node_y - end[1]
    run_y = [start[1] + shift + i * pitch for i in range(N_TURNS)]
    left, right = start[0], end[0]
    if abs(run_y[-1] - node_y) > dbu or abs(right - point["turn_length"] / 2) > dbu:
        raise ValueError("Meander placement differs from drum_lc_resonator")

    # Slits: ground lead, node end of the last run, and block boundaries
    reach = (pitch - w) / 2  # half the space between runs
    cuts = gf.kdb.Region()
    cuts += _box(
        left - 5.0 - CUT / 2,
        run_y[0] - reach,
        left - 5.0 + CUT / 2,
        run_y[0] + reach,
        dbu,
    )
    x_node = right - 3 * w
    cuts += _box(
        x_node - CUT / 2, node_y - reach, x_node + CUT / 2, node_y + reach, dbu
    )
    edges = segment_runs()
    for run in edges[1:-1]:
        # The connector from run - 1 to run is at the right end after an even run
        x0, x1 = (right - w, right) if (run - 1) % 2 == 0 else (left, left + w)
        y = (run_y[run - 1] + run_y[run]) / 2
        cuts += _box(x0 - 1.0, y - CUT / 2, x1 + 1.0, y + CUT / 2, dbu)

    o1, o2 = cell.ports["o1"], cell.ports["o2"]
    bounds = cell.dbbox()
    box = _box(
        bounds.left - GROUND_MARGIN,
        bounds.bottom - GROUND_MARGIN,
        bounds.right + GROUND_MARGIN,
        bounds.top + GROUND_MARGIN,
        dbu,
    )
    etch = cell.get_region(LAYER.M1_ETCH)
    span = feed_xs.width + 2 * feed_gap
    etch += _box(o1.dx - feed_gap, -span / 2, o1.dx, span / 2, dbu)
    etch += _box(o2.dx, -span / 2, o2.dx + feed_gap, span / 2, dbu)
    metal = (((box - etch) | cell.get_region(LAYER.M1_DRAW)) - cuts).merged()

    pad_x = right - point["coupling_pad_size"] / 2
    probes = {
        "node": (pad_x, pad_bottom + point["coupling_pad_size"] / 2),
        "feed": ((o1.dx + o2.dx) / 2, 0.0),
    }
    for k in range(SEGMENTS):
        probes[f"meander_{k + 1}"] = (0.0, run_y[(edges[k] + edges[k + 1] - 1) // 2])
    electrodes = {name: _pick(metal, *xy, dbu, name) for name, xy in probes.items()}
    ground = _pick(metal, bounds.left - GROUND_MARGIN / 2, 0.0, dbu, "ground")
    names = list(electrodes)
    for i, name in enumerate(names):
        if not (electrodes[name] & ground).is_empty():
            raise ValueError(f"{name} touches chip ground")
        for other in names[:i]:
            if not (electrodes[name] & electrodes[other]).is_empty():
                raise ValueError(f"Electrodes {name} and {other} are shorted")
    covered = ground.dup()
    for region in electrodes.values():
        covered += region
    if not (metal - covered).is_empty():
        raise ValueError("Floating metal would be grounded silently")

    membrane = cell.get_region(LAYER.VG_TOP).merged()
    if membrane.count() != 1:
        raise ValueError("Expected one drum membrane")
    disc = membrane.bbox().to_dtype(dbu)
    # The membrane must cover the bottom electrode, which belongs to node
    centre = gf.kdb.Region(
        gf.kdb.Box(
            round(disc.center().x / dbu),
            round(disc.center().y / dbu),
            round(disc.center().x / dbu) + 1,
            round(disc.center().y / dbu) + 1,
        )
    )
    if electrodes["node"].interacting(centre).is_empty():
        raise ValueError("The drum electrode is not part of node")
    return {
        "electrodes": {
            "node": electrodes["node"],
            "membrane": membrane,
            "feed": electrodes["feed"],
            **{n: electrodes[n] for n in names if n.startswith("meander_")},
            "ground": ground,
        },
        "membrane_box": (disc.left, disc.bottom, disc.right, disc.top),
        "cuts": cuts,
        "dbu": dbu,
    }


def _with_box(
    mesh: Callable[[Any], None], box: dict[str, float]
) -> Callable[[Any], None]:
    """Add a gmsh Box size field to gsim's 3D refinement fields."""

    def wrapped(sim: Any) -> None:
        import gmsh  # ruff: ignore[import-outside-top-level]
        from gsim.palace.mesh import (  # ruff: ignore[import-outside-top-level]
            gmsh_utils,
        )

        finalize = gmsh_utils.finalize_mesh_fields
        added = []

        def finalize_with_box(field_ids: list[int]) -> None:
            field = max(field_ids) + 1
            gmsh.model.mesh.field.add("Box", field)
            for key, value in box.items():
                gmsh.model.mesh.field.setNumber(field, key, value)
            added.append(field)
            finalize([*field_ids, field])

        gmsh_utils.finalize_mesh_fields = finalize_with_box
        try:
            mesh(sim)
        finally:
            gmsh_utils.finalize_mesh_fields = finalize
        if not added:
            raise RuntimeError(
                "gsim did not build size fields; the gap refinement was not applied"
            )

    return wrapped


def geometry(
    point: dict[str, float],
    settings: Settings,
    processes: int,
    *,
    conductors: Literal["sheet", "solid"] = CONDUCTORS,
    drum_mesh: float = DRUM_MESH,
) -> Geometry:
    """Stack the M1 nets, the membrane and silicon, with the gap refined."""
    if conductors not in {"sheet", "solid"}:
        raise ValueError(f"conductors must be 'sheet' or 'solid', got {conductors!r}")
    if not 0 < drum_mesh <= settings.near_mesh:
        raise ValueError("Require 0 < drum_mesh <= near_mesh")
    parts = layout(point)
    m1 = LAYER_STACK.layers["M1"].thickness
    top = LAYER_STACK.layers["VacuumGap_Top"]
    substrate = LAYER_STACK.layers["Substrate"].thickness
    gap = top.zmin - m1
    # Sheets sit at their bottom z, so the M1 sheet is at 0 and the membrane at the gap
    gap_bottom = 0.0 if conductors == "sheet" else m1
    membrane_z = gap_bottom + gap
    electrodes = {
        name: (region, 0.0, m1) for name, region in parts["electrodes"].items()
    }
    electrodes["membrane"] = (
        parts["electrodes"]["membrane"],
        membrane_z,
        membrane_z + top.thickness,
    )
    stack = stack_geometry(
        electrodes,
        terminals=TERMINALS,
        settings=settings,
        processes=processes,
        planar_conductors=conductors == "sheet",
        # The airbox supplies the vacuum, including the gap; gsim drops air dielectrics
        dielectrics=[
            {
                "name": "substrate",
                "zmin": -substrate,
                "zmax": 0.0,
                "material": "silicon",
            },
        ],
    )
    x0, y0, x1, y1 = parts["membrane_box"]
    box = {
        "VIn": drum_mesh,
        "VOut": settings.far_mesh,
        "XMin": x0,
        "XMax": x1,
        "YMin": y0,
        "YMax": y1,
        "ZMin": gap_bottom,
        "ZMax": membrane_z,
        "Thickness": DRUM_MESH_TRANSITION,
    }
    stack.mesh = _with_box(stack.mesh, box)
    stack.mesh_inputs |= {
        "conductors": conductors,
        "vacuum_gap_um": gap,
        "gap_refinement_box": box,
        "meander_cut_um": CUT,
        "segment_runs": segment_runs(),
    }
    return stack


RECIPE = Recipe(
    name="drum_lc_resonator_palace",
    script=Path(__file__),
    grid=GRID,
    terminals=TERMINALS,
    geometry=geometry,
    settings=SETTINGS,
    description="Default QPDK drum LC resonator with feedline, vacuum-gap membrane and segmented meander in a finite coplanar ground.",
    reference_ground="coplanar_chip_ground",
    provenance={
        "recipe_version": 1,
        "issue": 1047,
        "n_turns": N_TURNS,
        "terminal_weights": terminal_weights(),
        "segment_runs": segment_runs(),
        "conductors": CONDUCTORS,
        "drum_mesh_um": DRUM_MESH,
        "metal": "PEC; M1 nets on silicon, membrane at the VacuumGap_Top height, gap vacuum",
        "ground": f"Coplanar ground {GROUND_MARGIN:g} um beyond the cell, including the drum ring",
        "feed": "Feedline centre conductor cut open one CPW gap beyond the cell ports",
        "membrane": "VG_TOP disc without landing posts; exact for node quantities, C[membrane, membrane] misses the posts",
        "meander": f"{SEGMENTS} terminals of consecutive runs, separated by {CUT:g} um slits; also cut from ground and from the node",
        "reduction": "v = terminal_weights (node 1, membrane and feed 0, meander blocks at their mean wire position). C_c = -C[feed] @ v, C_d = -C[membrane] @ v, C_p = v @ C @ v - C_c - C_d",
        "limitations": "Lumped linear voltage profile on an electrically long meander; the driven meander recipe gives the distributed response. Substrate is bare silicon with no backside metal or package",
    },
)

CONVERGENCE: dict[str, tuple[dict, dict]] = {
    "nominal": ({}, {}),
    "drum_mesh_0.14": ({}, {"drum_mesh": 0.14}),
    "drum_mesh_0.035": ({}, {"drum_mesh": 0.035}),
    "near_mesh_2": ({"near_mesh": 2.0}, {}),
    "near_mesh_0.5": ({"near_mesh": 0.5}, {}),
    "order_2": ({"order": 2}, {}),
    "order_4": ({"order": 4}, {}),
    "domain_pad_1200": ({"domain_pad": 1200.0}, {}),
    "solid": ({}, {"conductors": "solid"}),
}
"""Single-point mesh study: settings changes and geometry keyword changes."""


def convergence(
    dry_run: bool = False,
    only: Annotated[list[str] | None, typer.Option()] = None,
    output: Path = Path("build/datasets/drum_convergence"),
    workdir: Path = Path("build/palace/drum_convergence"),
    executable: Path | None = None,
    sif: Path | None = None,
    processes: Annotated[int, typer.Option(min=1)] = 4,
    container_binary: str = "palace-x86_64.bin",
) -> None:
    """Solve the nominal point once per CONVERGENCE case and print C_c, C_p, C_d."""
    from qpdk.models.drum import (  # ruff: ignore[import-outside-top-level]
        drum_capacitances,
    )

    cases = {name: CONVERGENCE[name] for name in (only or CONVERGENCE)}
    for name, (settings_change, geometry_change) in cases.items():
        settings = replace(SETTINGS, **settings_change)
        logger.info(f"{name}: settings={settings}, geometry={geometry_change}")
        if dry_run:
            continue
        recipe = replace(
            RECIPE,
            name=f"{RECIPE.name}_{name}",
            geometry=partial(geometry, **geometry_change),
            provenance={
                **RECIPE.provenance,
                "convergence_case": name,
                "conductors": geometry_change.get("conductors", CONDUCTORS),
                "drum_mesh_um": geometry_change.get("drum_mesh", DRUM_MESH),
            },
        )
        data = generate(
            recipe,
            grid={axis: [value] for axis, value in NOMINAL.items()},
            output=output / name,
            workdir=workdir / name,
            executable=executable,
            sif=sif,
            processes=processes,
            container_binary=container_binary,
            settings=settings,
        )
        c_c, c_p, c_d = (
            float(c) * 1e15 for c in drum_capacitances(**NOMINAL, dataset=data)
        )
        typer.echo(f"{name:>16}: C_c={c_c:.4f} fF  C_p={c_p:.4f} fF  C_d={c_d:.4f} fF")


if __name__ == "__main__":
    if sys.argv[1:2] == ["convergence"]:
        del sys.argv[1]
        typer.run(convergence)
    else:
        run_cli(RECIPE)
