"""Tests for the hyperbolic lattice layout cells."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from klayout import db as kdb

from qpdk import PDK
from qpdk.cells.hyperbolic import (
    hyperbolic_edge_resonator,
    hyperbolic_lattice,
    hyperbolic_vertex_coupler,
)
from qpdk.hyperbolic import hyperbolic_tiling
from qpdk.tech import LAYER

PDK.activate()

_LAYERS = (LAYER.M1_DRAW, LAYER.M1_ETCH)


def _region(cell, trans=None) -> kdb.Region:
    region = kdb.Region()
    for layer in _LAYERS:
        region += kdb.Region(cell.kdb_cell.begin_shapes_rec(cell.kcl.layer(*layer)))
    region.merge()
    return region.transformed(trans) if trans is not None else region


@pytest.mark.parametrize("q", [3, 4, 5])
def test_vertex_coupler_ports(q: int) -> None:
    """Pins sit on a circle and point radially at ``rotation + 360 k / q``."""
    c = hyperbolic_vertex_coupler(q=q, rotation=17.0)
    assert len(c.ports) == q
    for k in range(q):
        port = c.ports[f"o{k + 1}"]
        x, y = port.center
        assert math.hypot(x, y) == pytest.approx(60.0, abs=2e-3)
        expected = (17.0 + 360 * k / q) % 360
        assert port.orientation % 360 == pytest.approx(expected, abs=1e-2)


def test_vertex_coupler_rejects_narrow_pins() -> None:
    """Leads wider than the pins are an error."""
    with pytest.raises(ValueError, match="narrower"):
        hyperbolic_vertex_coupler(q=12, pin_radius=15.0)
    with pytest.raises(ValueError, match="at least 2"):
        hyperbolic_vertex_coupler(q=1)


@pytest.mark.parametrize(
    ("length", "end", "start_angle", "end_angle"),
    [
        (4000.0, (1000.0, 0.0), 0.0, 0.0),
        (3000.0, (700.0, 300.0), 10.0, 40.0),
        (5000.0, (-400.0, 900.0), 95.0, 130.0),
        (1400.0, (1200.0, 0.0), 0.0, 0.0),
    ],
)
def test_edge_resonator_length(
    length: float,
    end: tuple[float, float],
    start_angle: float,
    end_angle: float,
) -> None:
    """The extruded centreline has the requested length and ends on its ports."""
    c = hyperbolic_edge_resonator(
        length=length, end=end, start_angle=start_angle, end_angle=end_angle
    )
    assert c.info["length"] == pytest.approx(length, abs=0.05)
    assert c.ports["o1"].center == pytest.approx((0.0, 0.0))
    assert c.ports["o2"].center == pytest.approx(end)
    assert c.ports["o1"].orientation % 360 == pytest.approx(
        (start_angle + 180) % 360, abs=1e-6
    )
    assert c.ports["o2"].orientation == pytest.approx(end_angle, abs=1e-6)


def test_edge_resonator_stays_in_region() -> None:
    """Metal and gaps stay inside the region shrunk by ``margin - half width``."""
    region = ((0.0, 0.0), (500.0, 600.0), (1000.0, 0.0), (500.0, -400.0))
    c = hyperbolic_edge_resonator(length=6000.0, region=region, margin=25.0)
    allowed = kdb.Region(
        kdb.DPolygon(list(itertools.starmap(kdb.DPoint, region))).to_itype(c.kcl.dbu)
    ).sized(-int((25.0 - 11.0 - 0.01) / c.kcl.dbu))
    # The end straights leave the region at its tips, so ignore 70 µm there.
    tips = kdb.Region()
    for x, y in ((0.0, 0.0), (1000.0, 0.0)):
        tips += kdb.Region(kdb.DBox(x - 70, y - 70, x + 70, y + 70).to_itype(c.kcl.dbu))
    outside = _region(c) - allowed - tips
    assert outside.is_empty()


def test_edge_resonator_too_long_raises() -> None:
    """A resonator that does not fit its region is an error."""
    with pytest.raises(ValueError, match="Region holds at most"):
        hyperbolic_edge_resonator(length=50_000.0)
    with pytest.raises(ValueError, match="cannot fit"):
        hyperbolic_edge_resonator(length=500.0)
    with pytest.raises(ValueError, match="too close"):
        hyperbolic_edge_resonator(length=200.0, end=(140.0, 0.0))


def test_edge_resonator_rejects_concave_region() -> None:
    """A non-convex region is an error rather than a silently different region."""
    region = ((0.0, 0.0), (500.0, 300.0), (1000.0, 0.0), (500.0, 600.0))
    with pytest.raises(ValueError, match="convex"):
        hyperbolic_edge_resonator(length=3000.0, region=region)


@pytest.fixture(scope="module")
def lattice():
    return hyperbolic_lattice()


def test_lattice_instances(lattice) -> None:
    """One coupler per vertex, one resonator per edge, one launcher per probe."""
    tiling = hyperbolic_tiling(7, 3, 1)
    names = [inst.name for inst in lattice.insts]
    assert sum(n.startswith("c") for n in names) == tiling.num_vertices
    assert sum(n.startswith("r") for n in names) == tiling.num_edges
    assert sum(n.startswith("launcher") for n in names) == 2
    assert sorted(p.name for p in lattice.ports) == ["o1", "o2"]
    for inst in lattice.insts:
        if inst.name.startswith("r"):
            assert inst.cell.info["length"] == pytest.approx(7460.0, abs=0.05)


def test_lattice_wiring_matches_tiling(lattice) -> None:
    """Every resonator end meets the coupler pin the tiling assigns to it."""
    tiling = hyperbolic_tiling(7, 3, 1)
    insts = {inst.name: inst for inst in lattice.insts}
    for e, (i, j) in enumerate(tiling.edges.tolist()):
        r = insts[f"r{e}"]
        for end, v in (("o1", i), ("o2", j)):
            k = tiling.pins[v][1].index(e)
            pin = insts[f"c{v}"].ports[f"o{k + 1}"]
            assert np.allclose(r.ports[end].center, pin.center, atol=1e-3)
            assert (r.ports[end].orientation - pin.orientation) % 360 == (
                pytest.approx(180.0, abs=1e-3)
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"p": 5, "q": 4},
        {"shells": 2, "disk_radius": 12000.0},
    ],
    ids=["7-3", "5-4", "7-3-two-shells"],
)
def test_lattice_has_no_overlaps(kwargs: dict) -> None:
    """Resonators keep at least 20 µm of ground between them away from couplers."""
    lattice = hyperbolic_lattice(**kwargs)
    tiling = hyperbolic_tiling(
        kwargs.get("p", 7), kwargs.get("q", 3), kwargs.get("shells", 1)
    )
    dbu = lattice.kcl.dbu
    regions = {inst.name: _region(inst.cell, inst.trans) for inst in lattice.insts}
    centres = {
        v: lattice.insts[f"c{v}"].dcplx_trans.disp for v in range(tiling.num_vertices)
    }
    centres = {v: (d.x, d.y) for v, d in centres.items()}
    vertices = {f"r{e}": {i, j} for e, (i, j) in enumerate(tiling.edges.tolist())}
    vertices |= {f"c{v}": {v} for v in range(tiling.num_vertices)}
    grow = int(10.0 / dbu)
    grown = {name: region.sized(grow) for name, region in regions.items()}
    for a, b in itertools.combinations(sorted(vertices), 2):
        if not grown[a].bbox().overlaps(grown[b].bbox()):
            continue
        shared = vertices[a] & vertices[b]
        if shared and not (a.startswith("r") and b.startswith("r")):
            continue  # a resonator and its own coupler
        overlap = grown[a] & grown[b]
        for v in shared:
            # Resonators on one coupler meet at its pins; spare a disk there.
            x, y = centres[v]
            overlap -= kdb.Region(
                kdb.DPolygon.ellipse(
                    kdb.DBox(x - 100, y - 100, x + 100, y + 100), 64
                ).to_itype(dbu)
            )
        assert overlap.is_empty(), (a, b)

    # Boundary stubs and probe feeds are drawn flat in the top cell.
    flat = kdb.Region()
    for layer in _LAYERS:
        flat += kdb.Region(lattice.kdb_cell.shapes(lattice.kcl.layer(*layer)))
    flat = flat.merged().sized(grow)
    for name, region in grown.items():
        if name.startswith("r"):
            assert (flat & region).is_empty(), name


def test_lattice_probe_pins_validated() -> None:
    """Out-of-range or repeated probe pins are an error."""
    with pytest.raises(ValueError, match="probe_pins"):
        hyperbolic_lattice(probe_pins=(0, 0))
    with pytest.raises(ValueError, match="probe_pins"):
        hyperbolic_lattice(probe_pins=(99,))


def test_lattice_too_small_raises() -> None:
    """Shrinking the disk until resonators no longer fit is an error."""
    with pytest.raises(ValueError, match="disk_radius"):
        hyperbolic_lattice(disk_radius=3000.0)


def test_edge_resonator_gap_below_one_meander_raises() -> None:
    """A length between the straight path and the shortest meander is an error."""
    with pytest.raises(ValueError, match="Cannot shorten"):
        hyperbolic_edge_resonator(length=1250.0, end=(1200.0, 0.0))
