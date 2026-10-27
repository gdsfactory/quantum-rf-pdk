"""Admission checks for the local Xmon island and readout paddle."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.datasets import (
    Dataset,
    GridInterpolator,
    check_maxwell,
    junction_model,
    maxwell_to_mutual,
)
from qpdk.models.datasets.validation import compare_capacitances


@pytest.fixture(scope="module")
def data() -> Dataset:
    return Dataset("xmon_readout_palace")


def test_shipped_grid_and_solver_diagnostics(data: Dataset) -> None:
    """The complete native grid includes coupling and solver diagnostics."""
    provenance = data.metadata.provenance
    assert not data.metadata.synthetic
    assert data.metadata.terminals == ("island", "readout")
    assert provenance["solver"] == "Palace"
    assert provenance["admission_status"] == "validated_local_electrostatic_model"
    assert provenance["validation"]["passed"]
    assert not provenance["validation"]["failed"]
    assert not provenance["validation"]["pending"]
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    assert grid.axis_names == (
        "arm_scale",
        "gap_width",
        "coupling_gap",
        "paddle_length",
    )
    coordinates = (
        [0.8, 1.0, 1.2],
        [8.0, 10.0, 12.0],
        [3.0, 4.0, 5.0, 7.5, 10.0],
        [30.0, 37.5, 45.0, 60.0, 75.0, 90.0],
    )
    for actual, expected in zip(grid.coords, coordinates, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert grid.values.shape == (3, 3, 5, 6, 2, 2)
    assert data.scan().collect().height == 270 * 11
    check_maxwell(grid.values)
    np.testing.assert_allclose(
        data.grid("mutual_capacitance", allow_missing=False).values,
        maxwell_to_mutual(grid.values),
        atol=1e-28,
    )
    residual = jnp.asarray(data.grid("solver_relative_residual").values)
    assert bool(jnp.all(jnp.isfinite(residual) & (residual >= 0) & (residual <= 1e-9)))


def test_independent_mesh_order_and_domain_references(data: Dataset) -> None:
    """Recompute every sensitivity test from its two native solves."""
    validation = data.metadata.provenance["validation"]
    checks = validation["numerical"]
    assert len(checks) == 15
    assert {row["control"] for row in checks} == {"fine-h", "hp4", "domain"}
    assert len({row["anchor"] for row in checks}) == 5
    for row in checks:
        result = compare_capacitances(
            row["candidate_maxwell_F"],
            row["reference_maxwell_F"],
            terminals=data.metadata.terminals,
            junction=("island", None),
            rtol=0.02,
            atol=1e-17,
        )
        assert result["passed"], (row["anchor"], row["control"], result)
        assert row["candidate_settings"] != row["reference_settings"]
        assert row["reference_source"]["provenance"]["solver"] == "Palace"


def test_interpolation_against_native_references(data: Dataset) -> None:
    """Check the published grid against direct solves throughout its domain."""
    validation = data.metadata.provenance["validation"]
    references = validation["interpolation"]
    assert len(references) == 135
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    lookup = GridInterpolator(grid)
    cells = set()
    boundaries = set()
    for row in references:
        result = compare_capacitances(
            lookup(**row["point"]),
            row["reference_maxwell_F"],
            terminals=data.metadata.terminals,
            junction=("island", None),
            rtol=0.02,
            atol=1e-17,
        )
        assert result["passed"], (row["label"], result)
        point = row["point"]
        for axis, coords in zip(grid.axis_names, grid.coords, strict=True):
            for boundary in (coords[0], coords[-1]):
                if point[axis] == boundary:
                    boundaries.add((axis, float(boundary)))
        if all(
            coords[0] < point[axis] < coords[-1] and point[axis] not in coords
            for axis, coords in zip(grid.axis_names, grid.coords, strict=True)
        ):
            cells.add(
                tuple(
                    int(np.searchsorted(coords, point[axis]) - 1)
                    for axis, coords in zip(grid.axis_names, grid.coords, strict=True)
                )
            )
    assert cells == set(product(range(2), range(2), range(4), range(5)))
    assert len(boundaries) == 8
    coverage = validation["held_out_coverage"]
    assert cells == {tuple(cell) for cell in coverage["tested_cell_indices"]}
    assert boundaries == {tuple(boundary) for boundary in coverage["boundaries_tested"]}


def test_internal_island_model_over_the_full_grid(data: Dataset) -> None:
    """Closing the junction preserves a lossless single external readout port."""
    grid = data.grid("maxwell_capacitance", allow_missing=False)
    model = junction_model(
        data,
        junction=("island", None),
        inductance=12e-9,
        junction_capacitance=1.5e-15,
        exposed_ports=("readout",),
    )
    points = jnp.asarray(list(product(*grid.coords)))
    frequencies = jnp.linspace(1e9, 10e9, 101)
    response = jax.jit(
        jax.vmap(
            lambda point: model(
                f=frequencies, **dict(zip(grid.axis_names, point, strict=True))
            )
        )
    )(points)
    assert set(response) == {("readout", "readout")}
    reflection = response["readout", "readout"]
    assert reflection.shape == (270, 101)
    assert bool(jnp.all(jnp.isfinite(reflection)))
    np.testing.assert_allclose(jnp.abs(reflection), 1, atol=1e-11)
    assert jnp.isnan(model(arm_scale=1.3)["readout", "readout"]).all()
