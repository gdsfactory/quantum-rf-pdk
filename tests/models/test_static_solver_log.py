"""Retained static-log validation uses models dependencies, never a solver backend."""

import builtins
from collections.abc import Callable
from typing import Any

import jax
import pytest

from qpdk.models.datasets.recipes import curate_double_pad
from qpdk.models.datasets.validation import static_solver_diagnostics

from . import test_double_pad_curation

packet = test_double_pad_curation.packet


@pytest.fixture(autouse=True)
def forbid_gsim_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a models-only installation even when local gsim is available."""
    original: Callable[..., Any] = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gsim" or name.startswith("gsim."):
            raise AssertionError("Static log validation must not import optional gsim")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def terminal(
    index: int,
    *,
    initial: str = "3e2",
    final: str = "1.5e-7",
    iterations: int = 25,
    outcome: str = "converged",
) -> str:
    """Build native-format terminal and separate auxiliary solver evidence."""
    return f"""  It {index}/2: Index = {index}, V = 1 V
  0 KSP residual norm {initial}
  {iterations} KSP residual norm {final}
  GMRES solver {outcome} in {iterations} iterations
  Updating solution error estimates
  0 KSP residual norm 1
  20 KSP residual norm 0.1
  CG solver did NOT converge
"""


def test_worst_terminal_metrics_exclude_auxiliary_failure() -> None:
    log = (
        "\x1b[31m"
        + terminal(1)
        + terminal(2, final="2.4e-7", iterations=30)
        + "\x1b[0m"
    )
    result = static_solver_diagnostics(log, terminals=2, tolerance=1e-9)
    assert result["solver_relative_residual"] == pytest.approx(8e-10)
    assert result["solver_iterations"] == 30


@pytest.mark.parametrize(
    "log",
    [
        "",
        "unindexed solver converged in 2 iterations",
        terminal(1),
        terminal(1) + terminal(1),
        terminal(2) + terminal(1),
        terminal(1) + terminal(3),
    ],
)
def test_exact_ordered_terminal_coverage(log: str) -> None:
    with pytest.raises(RuntimeError, match="Missing terminal"):
        static_solver_diagnostics(log, terminals=2, tolerance=1e-9)


@pytest.mark.parametrize(
    ("initial", "final"),
    [
        ("0", "0"),
        ("-1", "0"),
        ("inf", "1"),
        ("nan", "1"),
        ("1", "nan"),
        ("1", "inf"),
        ("1", "-1"),
        ("1", "invalid"),
        ("invalid", "1"),
        ("1e-320", "1"),
    ],
)
def test_invalid_or_overflowing_residuals(initial: str, final: str) -> None:
    with pytest.raises(RuntimeError, match="Missing solver convergence"):
        static_solver_diagnostics(
            terminal(1, initial=initial, final=final), terminals=1, tolerance=1e-9
        )


@pytest.mark.parametrize(
    "outcome", ["did NOT converge", "failed to converge", "diverged"]
)
def test_core_failure_cannot_be_hidden_by_success(outcome: str) -> None:
    log = terminal(1).replace(
        "Updating solution error estimates",
        f"GMRES solver {outcome}\nUpdating solution error estimates",
    )
    with pytest.raises(RuntimeError, match="Missing solver convergence"):
        static_solver_diagnostics(log, terminals=1, tolerance=1e-9)


@pytest.mark.parametrize(
    "damage",
    [
        "no_residuals",
        "one_residual",
        "no_success",
        "two_successes",
        "malformed_success",
    ],
)
def test_incomplete_or_ambiguous_core_solve(damage: str) -> None:
    log = terminal(1)
    match damage:
        case "no_residuals":
            log = log.replace("0 KSP residual norm 3e2", "").replace(
                "25 KSP residual norm 1.5e-7", ""
            )
        case "one_residual":
            log = log.replace("25 KSP residual norm 1.5e-7", "")
        case "no_success":
            log = log.replace("GMRES solver converged in 25 iterations", "")
        case "two_successes":
            log = log.replace(
                "Updating solution error estimates",
                "GMRES solver converged in 1 iteration\nUpdating solution error estimates",
            )
        case "malformed_success":
            log = log.replace(
                "converged in 25 iterations", "converged in unknown iterations"
            )
    with pytest.raises(RuntimeError, match="Missing solver convergence"):
        static_solver_diagnostics(log, terminals=1, tolerance=1e-9)


def test_relative_residual_tolerance_and_existing_roundoff_margin() -> None:
    for final in ("0", "1e-9", "1.000005e-9"):
        result = static_solver_diagnostics(
            terminal(1, initial="1", final=final), terminals=1, tolerance=1e-9
        )
        assert result["solver_relative_residual"] == pytest.approx(
            float(final), rel=0, abs=0
        )
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        static_solver_diagnostics(
            terminal(1, initial="1", final="1.00002e-9"), terminals=1, tolerance=1e-9
        )


def test_curator_native_proof_does_not_require_optional_backend(packet: dict) -> None:
    with jax.enable_x64():
        proof = curate_double_pad._native_run(**packet, legacy=True)
    assert proof["diagnostics_scope"] == "native solver log"
    assert proof["diagnostics"]["solver_relative_residual"] == pytest.approx(1e-10)
    assert proof["diagnostics"]["solver_iterations"] == 2
