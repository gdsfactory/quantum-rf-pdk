"""Square counts of meander_inductor corner styles and of nanowire_resonator.

The drawn square counts in :func:`qpdk.tech.meander_n_squares_drawn` are checked
against a first-order finite-element solve of Laplace's equation on the drawn
polygon, with the two wire ends as equipotential contacts.
"""

import gdsfactory as gf
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import shapely
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon

from qpdk import PDK
from qpdk.cells.inductor import meander_inductor
from qpdk.cells.nanowire import nanowire_resonator
from qpdk.tech import (
    CHAMFER_CORNER_SQUARES,
    LAYER,
    SHARP_CORNER_SQUARES,
    meander_n_squares_drawn,
)

PDK.activate()

CORNER_STYLES = ("sharp", "chamfer", "round", "optimal")


def _wire_xs(width: float) -> gf.CrossSection:
    return gf.cross_section.cross_section(width=width, layer=LAYER.NbTiN)


def _fem_squares(
    poly: Polygon, contact0: LineString, contact1: LineString, h: float
) -> float:
    """Resistance in squares between two boundary contacts of a polygon."""
    boundary = poly.exterior
    n_boundary = 4 * int(np.ceil(boundary.length / h))
    edge_pts = np.array([
        boundary.interpolate(t, normalized=True).coords[0]
        for t in np.linspace(0, 1, n_boundary, endpoint=False)
    ])
    minx, miny, maxx, maxy = poly.bounds
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx + h, h), np.arange(miny, maxy + h, h)
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid += np.random.default_rng(0).uniform(-0.15, 0.15, grid.shape) * h
    inner = grid[shapely.contains_xy(poly.buffer(-0.3 * h), grid[:, 0], grid[:, 1])]
    pts = np.unique(np.round(np.vstack([edge_pts, inner]), 9), axis=0)

    tris = Delaunay(pts).simplices
    centroids = pts[tris].mean(axis=1)
    tris = tris[shapely.contains_xy(poly, centroids[:, 0], centroids[:, 1])]
    p = pts[tris]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    keep = area > 1e-6 * h * h
    tris, p, area = tris[keep], p[keep], area[keep]

    x, y = p[:, :, 0], p[:, :, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    k_local = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (
        4 * area
    )[:, None, None]
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    k = sp.csr_matrix((k_local.ravel(), (rows, cols)), shape=(len(pts),) * 2)

    points = shapely.points(pts)
    on0 = contact0.distance(points) < 1e-6
    on1 = contact1.distance(points) < 1e-6
    v = np.zeros(len(pts))
    v[on1] = 1
    fixed = np.logical_or(on0, on1)
    free = np.logical_not(fixed)
    v[free] = spl.spsolve(k[free][:, free].tocsc(), -k[free][:, fixed] @ v[fixed])
    return float(1 / (v @ k @ v))


def test_straight_wire_is_length_over_width() -> None:
    """A single run has exactly length / width squares, for every style."""
    for style in CORNER_STYLES:
        c = meander_inductor(
            n_turns=1,
            turn_length=40.0,
            cross_section=_wire_xs(0.1),
            add_etch=False,
            corner_style=style,
        )
        assert c.info["n_squares_drawn"] == pytest.approx(400.0)
    assert meander_n_squares_drawn(1, 40.0, 0.1, 0.1) == pytest.approx(400.0)


def test_nanowire_resonator_straight_squares() -> None:
    """The straight nanowire counts wire_length / wire_width plus its two leads."""
    c = nanowire_resonator(wire_width=0.2, wire_length=30.0)
    assert c.info["n_squares"] == pytest.approx(152.0)


def test_sharp_corner_convention() -> None:
    """Sharp corners add the Hagedorn-Hall 0.559 squares to the straight runs."""
    w, g, length, n = 1.0, 1.0, 14.0, 3
    straight = n * length / w + (n - 1) * (g / w)
    expected = straight + 2 * (n - 1) * (SHARP_CORNER_SQUARES - 1)
    assert meander_n_squares_drawn(n, length, w, g, "sharp") == pytest.approx(expected)
    assert CHAMFER_CORNER_SQUARES > SHARP_CORNER_SQUARES


@pytest.mark.parametrize("corner_style", CORNER_STYLES)
def test_drawn_squares_match_fem(corner_style: str) -> None:
    """The square count in info matches a FEM solve on the drawn polygon."""
    c = meander_inductor(
        n_turns=3,
        turn_length=14.0,
        cross_section=_wire_xs(1.0),
        wire_gap=1.0,
        add_etch=False,
        corner_style=corner_style,
    )
    polys = [Polygon(p) for v in c.get_polygons_points().values() for p in v]
    poly = shapely.unary_union(polys).buffer(0)
    assert poly.geom_type == "Polygon"
    contacts = [
        LineString([(p.center[0], p.center[1] - 0.5), (p.center[0], p.center[1] + 0.5)])
        for p in c.ports
    ]
    fem = _fem_squares(poly, *contacts, h=0.1)
    assert fem == pytest.approx(c.info["n_squares_drawn"], rel=3e-3)


def test_chamfer_corner_value_from_fem() -> None:
    """FEM recovers the chamfer-minus-sharp corner difference, 0.577 - 0.559.

    The difference of two solves on the same mesh size cancels most of the
    discretisation error, so it resolves the 0.018 squares per corner that the
    absolute comparison above cannot.
    """
    n = 5
    fem = {}
    for style in ("sharp", "chamfer"):
        c = meander_inductor(
            n_turns=n,
            turn_length=6.0,
            cross_section=_wire_xs(1.0),
            wire_gap=1.0,
            add_etch=False,
            corner_style=style,
        )
        polys = [Polygon(p) for v in c.get_polygons_points().values() for p in v]
        poly = shapely.unary_union(polys).buffer(0)
        contacts = [
            LineString([
                (p.center[0], p.center[1] - 0.5),
                (p.center[0], p.center[1] + 0.5),
            ])
            for p in c.ports
        ]
        fem[style] = _fem_squares(poly, *contacts, h=0.05)
    per_corner = (fem["chamfer"] - fem["sharp"]) / (2 * (n - 1))
    assert per_corner == pytest.approx(
        CHAMFER_CORNER_SQUARES - SHARP_CORNER_SQUARES, abs=3e-3
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"corner_style": "round", "radius": 0.4}, "radius"),
        ({"corner_style": "round", "radius": 1.5}, "radius"),
        ({"corner_style": "optimal", "turn_length": 4.0}, "optimal"),
        ({"corner_style": "bevel"}, "corner_style"),
    ],
)
def test_invalid_corner_arguments(kwargs: dict, match: str) -> None:
    """Out-of-range radii, too-short optimal runs and unknown styles raise."""
    kwargs = {
        "n_turns": 3,
        "turn_length": 14.0,
        "cross_section": _wire_xs(1.0),
        "wire_gap": 1.0,
        "add_etch": False,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        meander_inductor(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wire_width": 0.02}, "wire_width"),
        ({"wire_width": 2.0}, "wire_width"),
        ({"n_turns": 2}, "n_turns"),
        ({"n_turns": 0}, "n_turns"),
    ],
)
def test_nanowire_resonator_rejects_invalid(kwargs: dict, match: str) -> None:
    """Widths outside 50 nm to 1 µm and even turn counts are rejected."""
    with pytest.raises(ValueError, match=match):
        nanowire_resonator(**kwargs)
