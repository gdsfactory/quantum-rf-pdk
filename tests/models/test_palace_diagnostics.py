"""Reject incomplete electrostatic solves before publishing dataset rows."""

import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data/_palace.py")
)["solver_diagnostics"]


def terminal(
    index: int,
    *,
    initial: str = "3.0e+02",
    final: str = "1.5e-07",
    converged: bool = True,
) -> str:
    """Palace's terminal and KSP log format, including its separate estimator solve."""
    convergence = (
        "GMRES solver converged in 25 iterations"
        if converged
        else "GMRES solver did not converge"
    )
    return f"""\nIt {index}/2: Index = {index}, V = 1.0e+00 V\n
0 KSP residual norm {initial}
25 KSP residual norm {final}
{convergence}
Updating solution error estimates
0 KSP residual norm 1.0e+00
25 KSP residual norm 1.0e-04
CG solver converged in 25 iterations
"""


def test_worst_terminal_and_estimator_solve_is_separate() -> None:
    result = CHECK(
        terminal(1) + terminal(2, final="2.4e-07"), terminals=2, tolerance=1e-9
    )
    assert result["solver_relative_residual"] == pytest.approx(8e-10)
    assert result["solver_iterations"] == 25


@pytest.mark.parametrize(
    ("log", "message"),
    [
        (terminal(1), "Missing terminal"),
        (terminal(1) + terminal(2, converged=False), "Missing solver convergence"),
        (terminal(1) + terminal(2, final="3.1e-07"), "exceeds tolerance"),
        (terminal(1) + terminal(2, final="nan"), "Missing solver convergence"),
        (terminal(1) + terminal(2, initial="inf"), "Missing solver convergence"),
        (terminal(1) + terminal(2, initial="0"), "Missing solver convergence"),
    ],
)
def test_incomplete_or_unconverged_terminal_is_rejected(log: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        CHECK(log, terminals=2, tolerance=1e-9)
