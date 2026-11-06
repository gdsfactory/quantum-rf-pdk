"""Admission checks for the shipped drum LC resonator extraction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import Dataset, check_maxwell
from qpdk.models.drum import (
    DRUM_DATASET,
    drum_capacitance_reduction,
    drum_capacitances,
)

NOMINAL = {
    "coupling_gap": 6.0,
    "coupling_pad_size": 20.0,
    "radius": 7.5,
    "turn_length": 300.0,
}


@pytest.fixture(scope="module")
def data() -> Dataset:
    return Dataset(DRUM_DATASET)


def test_shipped_grid_and_provenance(data: Dataset) -> None:
    """The published parts must be the admitted 54-point extraction."""
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert not data.metadata.synthetic
    assert grid.axis_names == (
        "coupling_gap",
        "coupling_pad_size",
        "radius",
        "turn_length",
    )
    assert grid.values.shape == (3, 3, 3, 2, 20, 20)
    assert data.metadata.terminals[:3] == ("node", "membrane", "feed")
    validation = data.metadata.provenance["validation"]
    assert validation["convergence"]["convergence_passed"]
    assert validation["convergence"]["segmentation"]["published"] == 17
    assert validation["verification"]["complete"]
    assert validation["verification"]["points"] == 54
    check_maxwell(grid.values)


def test_nominal_capacitances_are_physical() -> None:
    """The bundled nominal point must reduce to finite, ordered capacitances."""
    c_c, c_p, c_d = drum_capacitances(**NOMINAL)
    values = np.asarray([c_c, c_p, c_d], dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
    assert 3e-15 < c_c < 8e-15
    assert 6e-14 < c_p < 1.1e-13
    assert 1.5e-14 < c_d < 4e-14
    assert c_p > c_d > c_c


def test_nominal_matches_stored_grid_point(data: Dataset) -> None:
    """Interpolating a grid point must return that point's own reduction."""
    grid = data.grid("maxwell_capacitance")
    weights = data.metadata.provenance["terminal_weights"]
    stored = np.asarray(drum_capacitance_reduction(grid.values[1, 1, 1, 1], weights))
    np.testing.assert_allclose(np.asarray(drum_capacitances(**NOMINAL)), stored)


def test_couplings_move_the_right_way() -> None:
    """The interpolated surface must keep the monotonic trends of the physics."""
    assert (
        drum_capacitances(**{**NOMINAL, "coupling_gap": 3.0})[0]
        > drum_capacitances(**{**NOMINAL, "coupling_gap": 12.0})[0]
    )
    assert (
        drum_capacitances(**{**NOMINAL, "turn_length": 300.0})[1]
        > drum_capacitances(**{**NOMINAL, "turn_length": 200.0})[1]
    )
    assert (
        drum_capacitances(**{**NOMINAL, "radius": 10.0})[2]
        > drum_capacitances(**{**NOMINAL, "radius": 5.0})[2]
    )


def test_capacitances_jit():
    """The lookup must work when traced, as the SAX model calls it."""

    def evaluate(gap):
        return drum_capacitances(**{**NOMINAL, "coupling_gap": gap})

    c_c, c_p, c_d = jax.jit(evaluate)(jnp.asarray(6.0))
    assert np.all(np.isfinite(np.asarray([c_c, c_p, c_d])))
