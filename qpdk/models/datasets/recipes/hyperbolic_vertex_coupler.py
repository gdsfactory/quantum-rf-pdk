# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@c00764bc18cbcb0e58ccb5c7dfb880d550455d35",
#   "typer>=0.24,<1",
#   "gsim[meshwell] @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Palace capacitance of the hyperbolic vertex coupler, one dataset per ``q``.

The datasets are ``hyperbolic_vertex_coupler_q3_palace`` ({7,3} lattices) and
``hyperbolic_vertex_coupler_q4_palace`` ({5,4}). Terminals ``o1`` .. ``oq`` are
the pins of :func:`qpdk.cells.hyperbolic_vertex_coupler`, which sit at
``360 (k - 1) / q`` degrees. A matrix has a fixed size, so each ``q`` needs its
own dataset. Lead width and gap stay at the ``cpw`` cross-section, 10 and 6 um.

Each point is a zero-thickness PEC sheet on a silicon half-domain. Ground is a
disk of radius ``pin_radius + lead_length + 6 + 100`` um around the cell, cut
open 6 um beyond each lead end so that no pin touches it.

The raw matrix contains the capacitance of the 20 um leads. The SAX models
place the lumped coupler at the disk rim and add the leads as transmission
lines, so counting them here as well would double them. A coupon job with two
open strips of 100 and 200 um gives the line capacitance and the open-end
fringe. ``maxwell_capacitance`` subtracts that lead capacitance, so its
reference plane is the disk rim. ``maxwell_capacitance_with_leads`` keeps the
raw matrix.

Cloud runs use the ``dry-run``, ``mesh-only``, ``pilot``, ``submit`` and
``publish`` commands. ``pilot`` and ``submit`` only list their batches unless
``--execute`` is given; ``--execute`` is the only path to the cloud and uses at
most five jobs at a time. ``dry-run`` and ``mesh-only`` never leave this
machine. All geometry arguments are in micrometres.
"""

# Native solver imports need system libraries; keep previews independent of them.
# ruff: file-ignore[import-outside-top-level]

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Annotated, Any

import gdsfactory as gf
import numpy as np
import typer
from _cloud import (
    MAX_CONCURRENT,
    Job,
    Prepared,
    prepare_sim,
    read_complete,
    run_batches,
    sim_dir,
    unique_jobs,
)
from _experiment import Geometry, Settings

from qpdk import PDK, logger
from qpdk.cells import hyperbolic_vertex_coupler
from qpdk.models.datasets import Axis, Dataset, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.capacitance import check_maxwell, maxwell_to_mutual
from qpdk.models.datasets.generate import sweep, write
from qpdk.tech import LAYER

GRID = {
    "pin_radius": [25.0, 32.0, 40.0, 50.0, 60.0, 80.0],
    "pin_gap": [2.0, 4.0, 8.0],
    "ground_gap": [3.0, 6.0, 12.0],
}
PHASE_A_GRID = {**GRID, "pin_radius": [32.0, 40.0, 50.0]}
PHASE_B_GRID = {**GRID, "pin_radius": [25.0, 60.0, 80.0]}
# Midpoints of the gap axes in log coordinates, away from every grid node.
HELD_OUT_GRID = {
    3: {"pin_radius": 36.0, "pin_gap": 2.83, "ground_gap": 4.24},
    4: {"pin_radius": 45.0, "pin_gap": 5.66, "ground_gap": 8.49},
}
SETTINGS = Settings(near_mesh=0.7, far_mesh=21.0, domain_pad=200.0, order=3)
QS = (3, 4)
LEAD_LENGTH = 20.0
SLOT = 6.0  # the cpw gap: ground is cut open this far beyond each lead end
GROUND_MARGIN = 100.0
CIRCLE_SEGMENTS = 384  # divisible by 3 and 4, so the ground is q-fold symmetric
COUPON_LENGTHS = (100.0, 200.0)
COUPON_SPACING = 200.0
COUPON_COUPLING_TOLERANCE = 1e-3
# One Gmsh thread: with four, the same input gave tetrahedron counts that
# differed by 0.1% from run to run; with one it repeated exactly. A repeatable
# mesh also lets gsim's cloud cache, which hashes the mesh bytes, recognise a job.
MESH_THREADS = 1
MIN_EDGE_DBU = 5  # shorter polygon edges are slivers left by rounding
TERMINAL_CROSS_SECTION = "cpw"
RECIPE_VERSION = 1
WORKDIR = Path("build/palace/hyperbolic_vertex_coupler")
OUTPUT = Path("build/datasets")


def dataset_name(q: int) -> str:
    """Return the dataset name for ``q``-pin couplers."""
    return f"hyperbolic_vertex_coupler_q{q}_palace"


def terminals(q: int) -> tuple[str, ...]:
    """Return ``o1`` .. ``oq``, the cell ports in matrix order."""
    return tuple(f"o{k}" for k in range(1, q + 1))


def _cuts(ports: list[gf.Port], width: float, dbu: float) -> gf.kdb.Region:
    """Slots of length ``SLOT`` beyond each port, as wide as its ground clearance."""
    cuts = gf.kdb.Region()
    for port in ports:
        angle = math.radians(port.orientation)
        x, y = port.center
        path = gf.kdb.DPath(
            [
                gf.kdb.DPoint(x, y),
                gf.kdb.DPoint(x + SLOT * math.cos(angle), y + SLOT * math.sin(angle)),
            ],
            width + 2 * SLOT,
        )
        cuts.insert(path.polygon().to_itype(dbu))
    return cuts


def _select(metal: gf.kdb.Region, port: gf.Port, dbu: float) -> gf.kdb.Region:
    """Return the one polygon at the port, rejecting shorts and missing metal."""
    x, y = (round(value / dbu) for value in port.center)
    selected = metal.interacting(gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2)))
    if selected.count() != 1:
        raise ValueError(f"Port {port.name} does not select exactly one electrode")
    return selected


def _check_edges(metal: gf.kdb.Region) -> None:
    """Reject slivers: polygon edges of a few nanometres left by grid rounding."""
    for polygon in metal.each():
        for index in range(polygon.holes() + 1):
            contour = (
                polygon.each_point_hull()
                if index == 0
                else polygon.each_point_hole(index - 1)
            )
            points = list(contour)
            for a, b in zip(points, [*points[1:], points[0]], strict=True):
                if math.hypot(a.x - b.x, a.y - b.y) < MIN_EDGE_DBU:
                    raise ValueError(f"Sliver edge at ({a.x}, {a.y}) dbu")


def _assemble(
    footprint: gf.kdb.Region,
    device: gf.Component,
    ports: dict[str, gf.Port],
    *,
    width: float,
    expected: int,
    settings: Settings,
) -> Geometry:
    """Cut the footprint and select one electrode per named port.

    Returns:
        The metal as polygons on M1_DRAW with the named ports, and the domain.

    Raises:
        ValueError: If the metal is not ``expected`` separate polygons, a port
            does not select one, or an electrode touches ground.
    """
    dbu = device.kcl.dbu
    cuts = _cuts(list(device.ports), width, dbu)
    metal = (
        (footprint - device.get_region(LAYER.M1_ETCH) - cuts)
        | device.get_region(LAYER.M1_DRAW)
    ).merged()
    if metal.count() != expected:
        raise ValueError(f"Expected {expected} metal polygons, got {metal.count()}")
    _check_edges(metal)
    electrodes = {name: _select(metal, port, dbu) for name, port in ports.items()}
    selected = gf.kdb.Region()
    for region in electrodes.values():
        selected |= region
    ground = metal - selected
    if ground.count() != 1:
        raise ValueError(f"Expected one ground polygon, got {ground.count()}")
    reach = SLOT - 2 * dbu
    for name, port in ports.items():
        x, y = port.center
        guard = gf.kdb.Region(
            gf.kdb.DPolygon
            .ellipse(gf.kdb.DBox(2 * reach, 2 * reach), 256)
            .moved(x, y)
            .to_itype(dbu)
        )
        if not (guard & ground).is_empty():
            raise ValueError(f"{name} is closer than {SLOT} um to ground at its end")
    if not (selected & ground).is_empty():
        raise ValueError("An electrode touches ground")
    component = gf.Component()
    for polygon in metal.each():
        component.add_polygon(polygon.to_dtype(dbu), layer=LAYER.M1_DRAW)
    for name, port in ports.items():
        component.add_port(name, port=port, port_type="electrical")
    box = footprint.bbox()
    pad = settings.domain_pad
    return Geometry(
        component=component,
        bounds=(
            box.left * dbu - pad,
            box.bottom * dbu - pad,
            box.right * dbu + pad,
            box.top * dbu + pad,
        ),
        terminal_ports={name: name for name in ports},
    )


def geometry(
    point: dict[str, float],
    settings: Settings,
    _processes: int = 1,
    *,
    q: int,
    lead_length: float = LEAD_LENGTH,
    ground_margin: float = GROUND_MARGIN,
) -> Geometry:
    """Build the coupler metal with a circular ground and open lead ends.

    The cell draws pins and leads on M1_DRAW and the ground clearance on
    M1_ETCH; there is no ground polygon. The ground here is a disk of radius
    ``pin_radius + lead_length + 6 + ground_margin`` minus that clearance and
    minus a slot beyond each port.

    Returns:
        The component with ports ``o1`` .. ``oq`` and the domain bounds.
    """
    PDK.activate()
    cell = hyperbolic_vertex_coupler(
        q=q,
        rotation=0.0,
        pin_radius=point["pin_radius"],
        pin_gap=point["pin_gap"],
        ground_gap=point["ground_gap"],
        lead_length=lead_length,
        cross_section=TERMINAL_CROSS_SECTION,
    )
    dbu = cell.kcl.dbu
    radius = point["pin_radius"] + lead_length + SLOT + ground_margin
    footprint = gf.kdb.Region(
        gf.kdb.DPolygon.ellipse(
            gf.kdb.DBox(2 * radius, 2 * radius), CIRCLE_SEGMENTS
        ).to_itype(dbu)
    )
    width = gf.get_cross_section(TERMINAL_CROSS_SECTION).width
    ports = {name: cell.ports[name] for name in terminals(q)}
    return _assemble(
        footprint, cell, ports, width=width, expected=q + 1, settings=settings
    )


def coupon_geometry(settings: Settings) -> Geometry:
    """Two open CPW strips of different length in one ground, one terminal each.

    Both go in one job because an electrostatic solve needs two terminals. The
    strips are far enough apart that their mutual coupling is negligible.

    Returns:
        The component with terminals ``short`` and ``long``.
    """
    PDK.activate()
    device = gf.Component()
    strips = [
        device.add_ref(
            gf.components.straight(length=length, cross_section=TERMINAL_CROSS_SECTION)
        )
        for length in COUPON_LENGTHS
    ]
    strips[1].dmove((0.0, COUPON_SPACING))
    for ref, name in zip(strips, ("short", "long"), strict=True):
        device.add_port(f"{name}_start", port=ref.ports["o1"])
        device.add_port(f"{name}_end", port=ref.ports["o2"])
    dbu = device.kcl.dbu
    box = device.dbbox()
    margin = GROUND_MARGIN
    footprint = gf.kdb.Region(
        gf.kdb.DBox(
            box.left - SLOT - margin,
            box.bottom - margin,
            box.right + SLOT + margin,
            box.top + margin,
        ).to_itype(dbu)
    )
    width = gf.get_cross_section(TERMINAL_CROSS_SECTION).width
    ports = {"short": device.ports["short_start"], "long": device.ports["long_start"]}
    return _assemble(
        footprint, device, ports, width=width, expected=3, settings=settings
    )


def _versions() -> dict[str, str | None]:
    """Return solver-stack versions; ``None`` where a package is absent (previews)."""
    versions: dict[str, str | None] = {}
    for name in ("gsim", "meshwell", "gmsh", "gdsfactory"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _polygons(component: gf.Component) -> list[list[list[float]]]:
    """Return the M1_DRAW polygons of a component as point lists in um."""
    layer = gf.get_layer_tuple(LAYER.M1_DRAW)
    return [
        polygon.tolist() for polygon in component.get_polygons_points(by="tuple")[layer]
    ]


def _job(
    name: str,
    kind: str,
    geometry_: Geometry,
    settings: Settings,
    extra: dict[str, Any],
) -> Job:
    """Wrap a geometry as a cloud job whose identity is everything that is solved."""
    inputs = {
        "kind": kind,
        **extra,
        "bounds_um": list(geometry_.bounds),
        "terminals": {
            label: list(geometry_.component.ports[port].center)
            for label, port in geometry_.terminal_ports.items()
        },
        "polygons_um": _polygons(geometry_.component),
        "settings": asdict(settings),
        "solver": {
            "preconditioner": "BoomerAMG",
            "max_iterations": 1000,
            "initial_guess": False,
            "save_fields": 0,
        },
        "mesh": {"minimum_feature_elements": 4, "threads": MESH_THREADS},
        "versions": _versions(),
    }

    def prepare(run: Path) -> Prepared:
        return prepare_sim(
            geometry_,
            settings=settings,
            inputs=inputs,
            run_dir=run,
            threads=MESH_THREADS,
        )

    return Job(
        name=name,
        inputs=inputs,
        terminals=tuple(geometry_.terminal_ports),
        tolerance=settings.tolerance,
        prepare=prepare,
    )


def coupler_job(
    q: int,
    point: dict[str, float],
    *,
    settings: Settings = SETTINGS,
    lead_length: float = LEAD_LENGTH,
    ground_margin: float = GROUND_MARGIN,
) -> Job:
    """Return the cloud job for one coupler.

    The sweep grid is not part of the inputs, so a larger grid reuses the runs
    of a smaller one.
    """
    geometry_ = geometry(
        point, settings, q=q, lead_length=lead_length, ground_margin=ground_margin
    )
    short = {"pin_radius": "R", "pin_gap": "pg", "ground_gap": "gg"}
    label = "-".join(f"{short[key]}{value:g}" for key, value in point.items())
    return _job(
        f"q{q}-{label}-n{settings.near_mesh:g}f{settings.far_mesh:g}"
        f"-L{lead_length:g}-m{ground_margin:g}-p{settings.domain_pad:g}",
        "coupler",
        geometry_,
        settings,
        {
            "q": q,
            "point": point,
            "lead_length_um": lead_length,
            "ground_margin_um": ground_margin,
            "slot_um": SLOT,
        },
    )


def coupon_job(settings: Settings = SETTINGS) -> Job:
    """Return the lead-calibration job."""
    return _job(
        "coupon",
        "coupon",
        coupon_geometry(settings),
        settings,
        {
            "lengths_um": list(COUPON_LENGTHS),
            "spacing_um": COUPON_SPACING,
            "slot_um": SLOT,
        },
    )


def points(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Return every point of a grid, axes in ``GRID`` order."""
    return [dict(zip(grid, values, strict=True)) for values in product(*grid.values())]


def phase_grid(phase: str) -> dict[str, list[float]]:
    """Return the grid of phase ``a`` (radii 32-50), ``b`` (25, 60, 80) or ``all``."""
    grids = {"a": PHASE_A_GRID, "b": PHASE_B_GRID, "all": GRID}
    if phase not in grids:
        raise ValueError(f"Phase must be one of {sorted(grids)}, got {phase!r}")
    return grids[phase]


def production_jobs(q: int, phase: str, settings: Settings = SETTINGS) -> list[Job]:
    """Return the coupler jobs of a phase, or the held-out point (``held-out``)."""
    if phase == "held-out":
        return [coupler_job(q, HELD_OUT_GRID[q], settings=settings)]
    return [
        coupler_job(q, point, settings=settings) for point in points(phase_grid(phase))
    ]


def pilot_jobs() -> list[Job]:
    """Return the 13 pilot jobs: coupon, mesh convergence, truncation, leads.

    Jobs whose inputs equal a production job share its run directory.
    """
    centre = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 6.0}
    small = {"pin_radius": 25.0, "pin_gap": 2.0, "ground_gap": 3.0}
    wide = {"pin_radius": 40.0, "pin_gap": 4.0, "ground_gap": 12.0}
    levels = ((1.0, 30.0), (0.7, 21.0), (0.5, 15.0))

    def at(near: float, far: float, pad: float = SETTINGS.domain_pad) -> Settings:
        return replace(SETTINGS, near_mesh=near, far_mesh=far, domain_pad=pad)

    jobs = [coupon_job()]
    jobs += [coupler_job(3, centre, settings=at(*level)) for level in levels]
    jobs += [coupler_job(4, centre, settings=at(*level)) for level in levels[1:]]
    jobs += [coupler_job(3, small, settings=at(*level)) for level in levels[1:]]
    jobs += [coupler_job(3, wide, settings=at(*level)) for level in levels[1:]]
    jobs += [
        coupler_job(3, centre, settings=at(0.7, 10.0)),  # far-field ramp check
        coupler_job(3, centre, settings=at(0.7, 21.0, 400.0), ground_margin=200.0),
        coupler_job(3, centre, lead_length=60.0),
    ]
    return jobs


# --- Lead de-embedding ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Calibration:
    """Capacitance of an open CPW lead, from the coupon."""

    c_prime: float
    """Line capacitance to ground in F/m."""
    c_open: float
    """Capacitance of one open end, including the slot beyond it, in F."""
    coupon_key: str
    """Run identity of the coupon."""

    def lead(self, length_um: float) -> float:
        """Return the capacitance to ground of a lead with one open end, in F."""
        return self.c_prime * length_um * 1e-6 + self.c_open


def calibrate(matrix: np.ndarray, coupon_key: str = "") -> Calibration:
    """Extract line capacitance and open-end fringe from the coupon matrix.

    Uses each strip's capacitance to ground, the row sum of its Maxwell row;
    for decoupled strips that equals the diagonal entry. With strip lengths
    ``a < b`` and ground capacitances ``g_a``, ``g_b``:
    ``c' = (g_b - g_a) / (b - a)`` and ``C_open = (g_a - c' a) / 2``, since each
    strip has two open ends.

    Returns:
        The calibration.

    Raises:
        ValueError: If the matrix is not 2x2, the strips couple by more than
            ``COUPON_COUPLING_TOLERANCE`` of their diagonal, or the result is
            not positive.
    """
    c = np.asarray(matrix, dtype=float)
    if c.shape != (2, 2):
        raise ValueError(f"Coupon matrix must be 2x2, got {c.shape}")
    coupling = abs(c[0, 1]) / min(c[0, 0], c[1, 1])
    if coupling > COUPON_COUPLING_TOLERANCE:
        raise ValueError(f"Coupon strips couple by {coupling:.2e} of their diagonal")
    ground = c.sum(axis=1)
    short, long = COUPON_LENGTHS
    c_prime = (ground[1] - ground[0]) / ((long - short) * 1e-6)
    c_open = (ground[0] - c_prime * short * 1e-6) / 2
    if c_prime <= 0 or c_open < 0:
        raise ValueError(f"Non-physical lead calibration c'={c_prime}, C_open={c_open}")
    return Calibration(
        c_prime=float(c_prime), c_open=float(c_open), coupon_key=coupon_key
    )


def deembed(
    raw: np.ndarray, calibration: Calibration, lead_length: float = LEAD_LENGTH
) -> np.ndarray:
    """Remove the lead capacitance, moving the reference plane to the disk rim.

    Every pin carries one lead, which adds its capacitance to ground to the
    pin's diagonal entry only.

    Returns:
        ``raw - (c' L + C_open) I``.
    """
    raw = np.asarray(raw, dtype=float)
    return raw - calibration.lead(lead_length) * np.eye(raw.shape[-1])


def rotation_asymmetry(matrix: np.ndarray) -> float:
    """Return max |C - P C P^T| over the largest diagonal entry, P one pin step."""
    c = np.asarray(matrix, dtype=float)
    shift = np.roll(np.eye(c.shape[-1]), 1, axis=1)
    return float(np.abs(c - shift @ c @ shift.T).max() / np.diag(c).max())


# --- Publishing ----------------------------------------------------------------


def load_calibration(workdir: Path) -> Calibration:
    """Load the calibration from the finished coupon run.

    Returns:
        The calibration computed from the coupon matrix.

    Raises:
        FileNotFoundError: If the coupon has not completed.
    """
    job = coupon_job()
    record = read_complete(workdir, job)
    if record is None:
        raise FileNotFoundError("The coupon run is not complete; run the pilot first")
    return calibrate(np.array(record["maxwell_capacitance"]), job.key)


def publish(
    q: int,
    *,
    phase: str = "all",
    workdir: Path = WORKDIR,
    output: Path | None = None,
    validation: dict[str, Any] | None = None,
    settings: Settings = SETTINGS,
) -> Dataset:
    """Write the dataset from finished runs, de-embedding the leads.

    Returns:
        The written dataset.

    Raises:
        ValueError: If a point of the grid has no finished run.
    """
    from _palace import fingerprint

    PDK.activate()
    grid = phase_grid(phase)
    calibration = load_calibration(workdir)
    records: dict[tuple[float, ...], dict[str, Any]] = {}
    missing = []
    for point in points(grid):
        record = read_complete(workdir, coupler_job(q, point, settings=settings))
        if record is None:
            missing.append(point)
        else:
            records[tuple(point.values())] = record
    if missing:
        raise ValueError(
            f"{len(missing)} points have no finished run, e.g. {missing[0]}"
        )
    extra = [
        name
        for name in ("solver_relative_residual", "solver_iterations")
        if all(name in record["diagnostics"] for record in records.values())
    ]
    raw = {
        key: np.array(record["maxwell_capacitance"]) for key, record in records.items()
    }
    asymmetry = max(rotation_asymmetry(matrix) for matrix in raw.values())
    provenance = {
        **fingerprint(Path(__file__)),
        "recipe_version": RECIPE_VERSION,
        "issue": 1090,
        "settings": asdict(settings),
        "execution": "Palace electrostatic on the GDSFactory+ cloud through gsim, in batches of at most five jobs",
        "metal": "zero-thickness PEC sheets at the air/Si interface",
        "outer_boundary": "no boundary condition on the outer faces of the box (natural zero normal electric displacement)",
        "ground": (
            f"Disk of radius pin_radius + lead_length + {SLOT:g} + {GROUND_MARGIN:g} um "
            f"({CIRCLE_SEGMENTS}-gon) minus the cell clearance and a {SLOT:g} um slot beyond each lead end"
        ),
        "terminals": f"o_k is the pin at 360 (k - 1) / {q} degrees",
        "reference_plane": "disk rim; the lead capacitance is subtracted from every diagonal entry",
        "maxwell_capacitance_with_leads": "raw FEM matrix, reference plane at the lead ends",
        "calibration": {
            "line_capacitance_F_per_m": calibration.c_prime,
            "open_end_capacitance_F": calibration.c_open,
            "lead_length_um": LEAD_LENGTH,
            "lead_capacitance_F": calibration.lead(LEAD_LENGTH),
            "coupon_lengths_um": list(COUPON_LENGTHS),
            "coupon_run": calibration.coupon_key,
        },
        "lead_cross_section": {"width_um": 10.0, "gap_um": SLOT},
        "rotation_asymmetry_max": asymmetry,
        "validation": validation
        if validation is not None
        else {"status": "pending", "samples": []},
    }
    name = dataset_name(q)
    metadata = DatasetMetadata(
        name=name,
        description=(
            f"Palace Maxwell capacitance of the QPDK hyperbolic_vertex_coupler with q={q} pins "
            "on silicon, leads de-embedded to the disk rim."
        ),
        axes=tuple(Axis(name=axis, unit="um") for axis in GRID),
        variants=("cross_section",),
        terminals=terminals(q),
        reference_ground="chip_ground",
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="maxwell_capacitance_with_leads",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="mutual_capacitance",
                kind=QuantityKind.MUTUAL_CAPACITANCE,
                unit="F",
            ),
            *(
                Quantity(name=quantity, kind=QuantityKind.CIRCUIT_PARAMETER, unit="1")
                for quantity in ("fem_error_indicator_norm", *extra)
            ),
        ),
        provenance=provenance,
    )

    def solve_from_cache(**point: float | str) -> dict[str, Any]:
        key = tuple(point[axis] for axis in GRID)
        matrix = deembed(raw[key], calibration)
        check_maxwell(matrix)
        diagnostics = records[key]["diagnostics"]
        return {
            "maxwell_capacitance": matrix,
            "maxwell_capacitance_with_leads": raw[key],
            "mutual_capacitance": maxwell_to_mutual(matrix),
            **{
                quantity: diagnostics[quantity]
                for quantity in ("fem_error_indicator_norm", *extra)
            },
        }

    frame = sweep(metadata, solve_from_cache, grid, {"cross_section": ["cpw"]})
    result = write(output or OUTPUT / name, metadata, frame)
    for quantity in ("maxwell_capacitance", "maxwell_capacitance_with_leads"):
        result.grid(quantity, allow_missing=False, cross_section="cpw")
    return result


# --- Command line --------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__.split("\n\n")[0])
Q = Annotated[int, typer.Option(help="Pins per coupler: 3 ({7,3}) or 4 ({5,4}).")]
PHASE = Annotated[str, typer.Option(help="a, b, all or held-out.")]
WORK = Annotated[Path, typer.Option(help="Run directory.")]


def _q(q: int) -> int:
    """Return ``q`` if it is a supported pin count.

    Raises:
        typer.BadParameter: If ``q`` is not in ``QS``.
    """
    if q not in QS:
        raise typer.BadParameter(f"q must be one of {QS}")
    return q


def _status(job: Job, workdir: Path) -> str:
    """Return ``complete``, ``submitted`` or ``pending`` from the files of a run."""
    run = workdir / job.key
    if (run / "complete.json").is_file():
        return "complete"
    return "submitted" if (run / "job.json").is_file() else "pending"


def _plan(jobs: list[Job], workdir: Path) -> None:
    """Print the batches ``run_batches`` would use, without meshing or uploading."""
    unique = unique_jobs(jobs)
    todo = [job for job in unique if _status(job, workdir) != "complete"]
    typer.echo(
        f"{len(jobs)} jobs, {len(unique)} distinct, {len(unique) - len(todo)} complete, "
        f"{len(todo)} to run in {-(-len(todo) // MAX_CONCURRENT)} batches of at most {MAX_CONCURRENT}"
    )
    for start in range(0, len(todo), MAX_CONCURRENT):
        batch = todo[start : start + MAX_CONCURRENT]
        typer.echo(
            f"batch {start // MAX_CONCURRENT + 1}: "
            + ", ".join(f"{job.name} [{_status(job, workdir)}]" for job in batch)
        )


def _execute(jobs: list[Job], workdir: Path, retry_failed: bool) -> None:
    """Submit the jobs to the cloud in batches and print a summary.

    Raises:
        typer.Exit: With code 1 if any job failed.
    """
    from _cloud import GsimBackend

    report = run_batches(
        jobs, workdir=workdir, backend=GsimBackend(), retry_failed=retry_failed
    )
    typer.echo(
        f"reused {len(report.reused)}, reattached {len(report.reattached)}, "
        f"submitted {len(report.submitted)}, completed {len(report.completed)}, "
        f"failed {len(report.failed)}"
    )
    for name, error in report.failed.items():
        typer.echo(f"FAILED {name}: {error}")
    if report.failed:
        raise typer.Exit(1)


@app.command("dry-run")
def dry_run_command(q: Q = 3, phase: PHASE = "all", workdir: WORK = WORKDIR) -> None:
    """List the points and batches, building every geometry but no mesh."""
    jobs = production_jobs(_q(q), phase)
    logger.info(f"{dataset_name(q)}: phase {phase}, {len(jobs)} points")
    _plan(jobs, workdir)


@app.command("mesh-only")
def mesh_only_command(
    q: Q = 3,
    phase: PHASE = "a",
    workdir: WORK = WORKDIR,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    keep_mesh: Annotated[bool, typer.Option(help="Keep palace.msh.")] = False,
) -> None:
    """Mesh and validate the config locally; print tetrahedra per point.

    Nothing is uploaded. The mesh is deleted unless ``--keep-mesh`` is given,
    since a full grid of meshes takes several gigabytes.
    """
    jobs = unique_jobs(production_jobs(_q(q), phase))[:limit]
    typer.echo("tetrahedra  valid  job")
    for job in jobs:
        run = workdir / job.key
        try:
            prepared = job.prepare(run)
            typer.echo(f"{prepared.tetrahedra!s:>10}  yes    {job.name}")
        except Exception as error:
            typer.echo(f"{'-':>10}  NO     {job.name}: {error}")
        if not keep_mesh:
            (sim_dir(run) / "palace.msh").unlink(missing_ok=True)


@app.command("pilot")
def pilot_command(
    workdir: WORK = WORKDIR,
    execute: Annotated[bool, typer.Option(help="Submit to the cloud.")] = False,
    retry_failed: bool = False,
) -> None:
    """List (or with --execute, run) the 13 pilot jobs."""
    jobs = pilot_jobs()
    _plan(jobs, workdir)
    if execute:
        _execute(jobs, workdir, retry_failed)


@app.command("submit")
def submit_command(
    q: Q = 3,
    phase: PHASE = "a",
    workdir: WORK = WORKDIR,
    execute: Annotated[bool, typer.Option(help="Submit to the cloud.")] = False,
    retry_failed: bool = False,
) -> None:
    """List (or with --execute, run) the jobs of a phase."""
    jobs = production_jobs(_q(q), phase)
    _plan(jobs, workdir)
    if execute:
        _execute(jobs, workdir, retry_failed)


@app.command("publish")
def publish_command(
    q: Q = 3,
    phase: PHASE = "all",
    workdir: WORK = WORKDIR,
    output: Path | None = None,
    validation: Annotated[
        Path | None, typer.Option(help="JSON file with the validation block.")
    ] = None,
) -> None:
    """Write the dataset from finished runs."""
    import json

    publish(
        _q(q),
        phase=phase,
        workdir=workdir,
        output=output,
        validation=json.loads(validation.read_text("utf-8")) if validation else None,
    )


if __name__ == "__main__":
    app()
