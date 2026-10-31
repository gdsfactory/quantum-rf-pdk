"""Validation must resolve small couplings and both readout boundaries."""

from typing import Any

import jax.numpy as jnp
import pytest

from qpdk.models.datasets.validation import compare_capacitances


@pytest.mark.parametrize("grounded", [False, True])
def test_weak_coupling_is_not_hidden_by_large_diagonal(grounded: bool) -> None:
    reference = (
        jnp.asarray([[80.0, -30.0, -0.1], [-30.0, 70.0, -0.01], [-0.1, -0.01, 10.0]])
        * 1e-15
    )
    changed = reference.at[0, 2].multiply(2).at[2, 0].multiply(2)
    report = compare_capacitances(
        changed,
        reference,
        terminals=("a", "b", "readout"),
        junction=("a", None if grounded else "b"),
    )
    assert not report["passed"]
    assert report["errors"]["matrix"]["allowance_fraction"] > 1
    assert report["errors"]["qubit_readout_grounded"]["allowance_fraction"] == 0
    assert report["errors"]["qubit_readout_floating"]["max_absolute_F"] > 0


def test_ground_sum_has_its_own_error_budget() -> None:
    reference = jnp.asarray([[100.0, -99.0], [-99.0, 100.0]]) * 1e-15
    changed = reference + jnp.eye(2) * 0.1e-15
    report = compare_capacitances(
        changed, reference, terminals=("a", "b"), junction=("a", "b")
    )
    assert report["errors"]["matrix"]["allowance_fraction"] < 1
    assert report["errors"]["ground"]["allowance_fraction"] > 1
    assert not report["passed"]


@pytest.mark.parametrize("junction", [("a", None), ("a", "b")])
def test_same_solve_passes(junction: tuple[str, str | None]) -> None:
    matrix = jnp.asarray([[80.0, -30.0], [-30.0, 70.0]]) * 1e-15
    assert compare_capacitances(
        matrix, matrix, terminals=("a", "b"), junction=junction
    )["passed"]


@pytest.mark.parametrize("change", ["shape", "junction", "tolerance", "nan"])
def test_reject_bad_references(change: str) -> None:
    matrix = jnp.eye(2) * 1e-15
    kwargs: dict[str, Any] = {"terminals": ("a", "b"), "junction": ("a", "b")}
    if change == "shape":
        matrix = matrix[:1, :1]
    elif change == "junction":
        kwargs["junction"] = ("a", "missing")
    elif change == "tolerance":
        kwargs["atol"] = 0
    else:
        matrix = matrix.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match=r"Matrices|Junction|tolerance|non-finite"):
        compare_capacitances(matrix, matrix, **kwargs)
