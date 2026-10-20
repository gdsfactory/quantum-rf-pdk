"""Mesh acceptance checks must include weak coupling and equal-trace symmetry."""

import runpy
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import pytest

EXPERIMENT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "qpdk/models/datasets/data/cpw_coupling.py"
    )
)


def solve_sequence(matrices: list) -> tuple[list[object], Callable[[object], dict]]:
    """Supply independently computed mesh results and keep the requested settings."""
    settings: list[object] = []
    iterator = iter(matrices)

    def solve(config: object) -> dict:
        settings.append(config)
        return {
            "maxwell_capacitance": jnp.asarray(next(iterator)),
            "solver_iterations": 20.0,
        }

    return settings, solve


def test_refines_past_a_mesh_outlier_and_records_the_accepted_mesh() -> None:
    settings, solve = solve_sequence([
        [[5.80547e-16, -3.99069e-18], [-3.99069e-18, 5.43116e-16]],
        [[5.22439e-16, -3.94299e-18], [-3.94299e-18, 5.22366e-16]],
        [[5.20849e-16, -3.93414e-18], [-3.93414e-18, 5.20809e-16]],
    ])
    result = EXPERIMENT["converge_mesh"](solve, EXPERIMENT["SETTINGS"])
    assert len(settings) == 3
    assert result["mesh_refinement_level"] == 2
    assert result["mesh_relative_change"] == pytest.approx(
        0.003052754038900354, rel=1e-3
    )
    assert result["mesh_near_size"] == pytest.approx(0.07e-6)
    assert result["mesh_far_size"] == pytest.approx(14e-6)
    assert result["solver_iterations"] == 20
    assert result["maxwell_capacitance"][0, 0] == pytest.approx(
        5.20849e-16, rel=1e-8, abs=0
    )


def test_small_mutual_capacitance_is_checked_separately() -> None:
    _, solve = solve_sequence([
        [[5e-16, -4e-18], [-4e-18, 5e-16]],
        [[5e-16, -3.8e-18], [-3.8e-18, 5e-16]],
    ])
    with pytest.raises(RuntimeError, match="did not converge"):
        EXPERIMENT["converge_mesh"](solve, EXPERIMENT["SETTINGS"], max_refinements=1)


def test_small_change_does_not_excuse_asymmetric_traces() -> None:
    _, solve = solve_sequence([[[5e-16, -4e-18], [-4e-18, 5.4e-16]]] * 2)
    with pytest.raises(RuntimeError, match="did not converge"):
        EXPERIMENT["converge_mesh"](solve, EXPERIMENT["SETTINGS"], max_refinements=1)


def test_nonfinite_mesh_results_are_rejected() -> None:
    _, solve = solve_sequence([[[jnp.nan, -4e-18], [-4e-18, 5e-16]]])
    with pytest.raises(RuntimeError, match="finite 2x2"):
        EXPERIMENT["converge_mesh"](solve, EXPERIMENT["SETTINGS"])


def test_requested_refinement_continues_past_an_early_small_change() -> None:
    settings, solve = solve_sequence([
        [[5e-16, -4e-18], [-4e-18, 5e-16]],
        [[5e-16, -3.99e-18], [-3.99e-18, 5e-16]],
        [[5e-16, -3.8e-18], [-3.8e-18, 5e-16]],
        [[5e-16, -3.79e-18], [-3.79e-18, 5e-16]],
    ])
    result = EXPERIMENT["converge_mesh"](
        solve, EXPERIMENT["SETTINGS"], min_refinements=2
    )
    assert len(settings) == 4
    assert result["mesh_refinement_level"] == 3
    assert result["mesh_relative_change"] < 0.01
    assert result["maxwell_capacitance"][0, 1] == pytest.approx(
        -3.79e-18, rel=1e-8, abs=0
    )


@pytest.mark.parametrize("min_refinements", [0, 5])
def test_invalid_minimum_refinements(min_refinements: int) -> None:
    _, solve = solve_sequence([])
    with pytest.raises(ValueError, match="min refinements"):
        EXPERIMENT["converge_mesh"](
            solve, EXPERIMENT["SETTINGS"], min_refinements=min_refinements
        )


@pytest.mark.parametrize(("tolerance", "max_refinements"), [(0, 4), (1, 4), (0.01, 0)])
def test_invalid_acceptance_settings(tolerance: float, max_refinements: int) -> None:
    _, solve = solve_sequence([])
    with pytest.raises(ValueError, match="mesh tolerance"):
        EXPERIMENT["converge_mesh"](
            solve,
            EXPERIMENT["SETTINGS"],
            tolerance=tolerance,
            max_refinements=max_refinements,
        )
