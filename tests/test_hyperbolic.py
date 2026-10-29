"""Tests for the hyperbolic {p,q} tiling generator in :mod:`qpdk.hyperbolic`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from qpdk.hyperbolic import (
    _reflect,
    central_polygon_radius,
    geodesic_tangent_angle,
    hyperbolic_tiling,
)


@st.composite
def schlafli(draw: st.DrawFn) -> tuple[int, int]:
    """A hyperbolic {p,q}: :math:`(p-2)(q-2) > 4`."""
    p = draw(st.integers(3, 10))
    q = draw(st.integers(3, 8))
    assume((p - 2) * (q - 2) > 4)
    return p, q


_disk_point = st.builds(
    lambda r, phi: r * np.exp(1j * phi),
    st.floats(0.0, 0.95),
    st.floats(0.0, 2 * np.pi),
)


def _hyperbolic_distance(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Poincaré-disk distance between ``z`` and ``w``.

    Returns:
        Distances with curvature :math:`-1`.
    """
    num = 2 * np.abs(z - w) ** 2
    return np.arccosh(1 + num / ((1 - np.abs(z) ** 2) * (1 - np.abs(w) ** 2)))


@pytest.mark.parametrize(
    ("shells", "vertices", "edges", "faces"),
    [(0, 7, 7, 1), (1, 35, 42, 8), (2, 112, 140, 29)],
)
def test_heptagonal_flake_counts(
    shells: int, vertices: int, edges: int, faces: int
) -> None:
    """The {7,3} flake sizes match a hand count and Kollár et al.'s 140 resonators."""
    tiling = hyperbolic_tiling(7, 3, shells)
    assert tiling.num_vertices == vertices
    assert tiling.num_edges == edges
    assert len(tiling.faces) == faces


@pytest.mark.parametrize(("p", "q", "shells"), [(7, 3, 2), (8, 3, 2), (5, 4, 1)])
def test_tiling_is_valid(p: int, q: int, shells: int) -> None:
    """Every vertex has at most ``q`` edges and every face is a closed ``p``-cycle."""
    tiling = hyperbolic_tiling(p, q, shells)
    assert tiling.degree.max() == q
    assert tiling.degree.min() >= 2
    assert np.all(np.abs(tiling.vertices) < 1)
    edge_set = {tuple(e) for e in tiling.edges.tolist()}
    for face in tiling.faces:
        assert len(set(face)) == p
        for k in range(p):
            i, j = face[k], face[(k + 1) % p]
            assert (min(i, j), max(i, j)) in edge_set


def test_central_polygon_radius() -> None:
    """The central heptagon of {7,3} sits at the textbook radius."""
    assert central_polygon_radius(7, 3) == pytest.approx(0.300742618746379)
    tiling = hyperbolic_tiling(7, 3, 0)
    np.testing.assert_allclose(np.abs(tiling.vertices), 0.300742618746379)


@pytest.mark.parametrize(("p", "q"), [(6, 3), (4, 4), (3, 6), (2, 7)])
def test_rejects_non_hyperbolic(p: int, q: int) -> None:
    """Euclidean and spherical tilings are rejected."""
    with pytest.raises(ValueError, match="not a hyperbolic tiling"):
        hyperbolic_tiling(p, q, 1)


@pytest.mark.parametrize("shells", [1, 2])
def test_line_graph_flat_band(shells: int) -> None:
    r"""The line graph has a flat band at :math:`-2` with multiplicity :math:`E - V`.

    For a connected non-bipartite graph the incidence matrix has full row
    rank, so :math:`B^\mathsf{T}B = A_L + 2I` has a null space of dimension
    :math:`E - V`.
    """
    tiling = hyperbolic_tiling(7, 3, shells)
    eigenvalues = np.linalg.eigvalsh(tiling.line_graph_adjacency)
    assert eigenvalues.min() == pytest.approx(-2)
    flat = np.isclose(eigenvalues, -2, atol=1e-9).sum()
    assert flat == tiling.num_edges - tiling.num_vertices


def test_line_graph_neighbours() -> None:
    """Edge resonators couple to exactly the other edges at their two vertices."""
    tiling = hyperbolic_tiling(7, 3, 1)
    a_l = tiling.line_graph_adjacency
    expected = tiling.degree[tiling.edges].sum(axis=1) - 2
    np.testing.assert_array_equal(a_l.sum(axis=1), expected)
    assert np.all(np.diag(a_l) == 0)
    np.testing.assert_array_equal(a_l, a_l.T)


def test_edge_faces() -> None:
    """Interior edges border two faces and boundary edges one."""
    tiling = hyperbolic_tiling(7, 3, 1)
    counts = np.array([len(faces) for faces in tiling.edge_faces])
    assert set(counts.tolist()) == {1, 2}
    for e, faces in enumerate(tiling.edge_faces):
        i, j = tiling.edges[e]
        for face in faces:
            assert {i, j} <= set(tiling.faces[face])


def test_distance_weighted_adjacency() -> None:
    """Central edges have weight 1 and outer (shorter) edges couple more strongly."""
    tiling = hyperbolic_tiling(7, 3, 1)
    a = tiling.distance_weighted_adjacency()
    np.testing.assert_allclose(a, a.T)
    assert a[0, 1] == pytest.approx(1.0)
    assert a.max() > 1.0
    np.testing.assert_array_equal(a > 0, tiling.adjacency > 0)


def test_cached_tiling_is_read_only() -> None:
    """Tilings are cached and shared, so their arrays cannot be modified."""
    tiling = hyperbolic_tiling(7, 3, 1)
    for a in (tiling.vertices, tiling.edges, tiling.incidence, tiling.adjacency):
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 0


@settings(deadline=None)
@given(pq=schlafli(), shells=st.integers(0, 2))
def test_flake_is_a_valid_tiling_patch(pq: tuple[int, int], shells: int) -> None:
    """Any hyperbolic flake is a disk patch of regular, edge-sharing ``p``-gons.

    Checks Euler's formula :math:`V - E + F = 1` for a disk, vertex degrees,
    closed ``p``-gon faces and a symmetric loop-free adjacency.
    """
    p, q = pq
    tiling = hyperbolic_tiling(p, q, shells)
    faces = len(tiling.faces)
    assert tiling.num_vertices - tiling.num_edges + faces == 1
    assert np.all(np.abs(tiling.vertices) < 1)
    assert tiling.degree.min() >= 2
    assert tiling.degree.max() <= q
    a = tiling.adjacency
    np.testing.assert_array_equal(a, a.T)
    assert np.all(np.diag(a) == 0)
    for face in tiling.faces:
        assert len(set(face)) == p
        for k in range(p):
            assert a[face[k], face[(k + 1) % p]] == 1


@settings(deadline=None)
@given(pq=schlafli(), shells=st.integers(0, 2))
def test_flake_edges_have_equal_hyperbolic_length(
    pq: tuple[int, int], shells: int
) -> None:
    """All tiling edges are congruent, so their hyperbolic lengths agree."""
    tiling = hyperbolic_tiling(*pq, shells)
    i, j = tiling.edges.T
    d = _hyperbolic_distance(tiling.vertices[i], tiling.vertices[j])
    np.testing.assert_allclose(d, d[0], rtol=1e-7)


@settings(deadline=None)
@given(pq=schlafli(), shells=st.integers(1, 2))
def test_interior_vertex_angles(pq: tuple[int, int], shells: int) -> None:
    r"""Geodesics leave a full-degree vertex :math:`2\pi/q` apart (conformality)."""
    p, q = pq
    tiling = hyperbolic_tiling(p, q, shells)
    for v in np.flatnonzero(tiling.degree == q):
        z = complex(tiling.vertices[v])
        angles = np.sort([
            geodesic_tangent_angle(z, complex(tiling.vertices[u]))
            for u in np.flatnonzero(tiling.adjacency[v])
        ])
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        np.testing.assert_allclose(gaps, 2 * np.pi / q, atol=1e-7)


@settings(deadline=None)
@given(pq=schlafli(), shells=st.integers(0, 2))
def test_line_graph_spectrum_bounded_below(pq: tuple[int, int], shells: int) -> None:
    r"""The line-graph adjacency is :math:`B^\mathsf{T}B - 2I`, so :math:`\lambda \ge -2`."""
    tiling = hyperbolic_tiling(*pq, shells)
    a_l = tiling.line_graph_adjacency
    b = tiling.incidence
    np.testing.assert_array_equal(a_l + 2 * np.eye(tiling.num_edges), b.T @ b)
    assert np.linalg.eigvalsh(a_l).min() >= -2 - 1e-9


@given(p=st.integers(2, 12), q=st.integers(2, 12))
def test_central_polygon_radius_domain(p: int, q: int) -> None:
    """The radius exists, inside the unit disk, exactly for hyperbolic {p,q}."""
    if p >= 3 and q >= 3 and (p - 2) * (q - 2) > 4:
        assert 0 < central_polygon_radius(p, q) < 1
    else:
        with pytest.raises(ValueError, match="not a hyperbolic tiling"):
            central_polygon_radius(p, q)


@given(a=_disk_point, b=_disk_point, z=_disk_point, w=_disk_point)
def test_reflection_is_a_hyperbolic_isometry(
    a: complex, b: complex, z: complex, w: complex
) -> None:
    """Reflection across a geodesic is an involution that preserves distance."""
    assume(abs(a - b) > 1e-3)
    pts = np.array([z, w])
    image = _reflect(pts, a, b)
    np.testing.assert_allclose(_reflect(image, a, b), pts, atol=1e-9)
    np.testing.assert_allclose(_reflect(np.array([a, b]), a, b), [a, b], atol=1e-9)
    np.testing.assert_allclose(
        _hyperbolic_distance(image[0], image[1]),
        _hyperbolic_distance(pts[0], pts[1]),
        rtol=1e-6,
        atol=1e-9,
    )
