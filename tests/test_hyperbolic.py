"""Tests for the hyperbolic {p,q} tiling generator in :mod:`qpdk.hyperbolic`."""

from __future__ import annotations

import numpy as np
import pytest

from qpdk.hyperbolic import central_polygon_radius, hyperbolic_tiling


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
