"""Independent accuracy checks for electrostatic qubit datasets."""

import re
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike

from qpdk.models.datasets.capacitance import (
    check_maxwell,
    differential_capacitance,
    maxwell_to_mutual,
)
from qpdk.models.datasets.metadata import QuantityKind
from qpdk.models.datasets.table import to_grid, validate

if TYPE_CHECKING:
    from qpdk.models.datasets.metadata import DatasetMetadata


def static_solver_diagnostics(
    log: str, *, terminals: int, tolerance: float
) -> dict[str, float]:
    """Check retained Palace static-solve logs without loading a solver backend.

    Residual ratios assume zero initial guesses, enforced by the caller's native
    config checks. Auxiliary estimator solves do not describe terminal convergence.

    Returns:
        Worst terminal final/initial residual ratio and iteration count.

    Raises:
        RuntimeError: If terminal solves are missing or fail the requested tolerance.
    """
    cleaned = (
        "\n".join(
            re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
            for line in log.splitlines()
        )
        + "\n"
    )
    blocks = re.split(
        r"^It \d+/\d+:\s+Index\s*=\s*(\d+)[^\n]*\n",
        cleaned,
        flags=re.MULTILINE,
    )
    metrics = []
    for index, block in zip(blocks[1::2], blocks[2::2], strict=True):
        field_log = block.split("Updating solution error estimates", 1)[0]
        iterations = re.findall(r"solver converged in (\d+) iterations?\b", field_log)
        try:
            residuals = [
                float(value)
                for value in re.findall(r"KSP residual norm (\S+)", field_log)
            ]
        except ValueError as error:
            raise RuntimeError("Missing solver convergence in Palace log") from error
        if (
            re.search(
                r"did not converge|failed to converge|diverged",
                field_log,
                re.IGNORECASE,
            )
            or len(iterations) != 1
            or len(residuals) < 2
            or not all(isfinite(value) and value >= 0 for value in residuals)
            or residuals[0] <= 0
        ):
            raise RuntimeError("Missing solver convergence in Palace log")
        ratio = residuals[-1] / residuals[0]
        if not isfinite(ratio):
            raise RuntimeError("Missing solver convergence in Palace log")
        metrics.append((int(index), int(iterations[0]), ratio))
    if [index for index, _, _ in metrics] != list(range(1, terminals + 1)):
        raise RuntimeError("Missing terminal solves in Palace log")
    ratio = max(ratio for _, _, ratio in metrics)
    if ratio > tolerance * 1.00001:
        raise RuntimeError(f"Solver residual {ratio:g} exceeds tolerance {tolerance:g}")
    return {
        "solver_relative_residual": ratio,
        "solver_iterations": float(max(iterations for _, iterations, _ in metrics)),
    }


def compare_capacitances(
    actual: ArrayLike,
    reference: ArrayLike,
    *,
    terminals: Sequence[str],
    junction: tuple[str, str | None],
    rtol: float = 0.02,
    atol: float = 1e-17,
) -> dict[str, Any]:
    """Check matrix entries, ground sums and qubit charge energy against a solve.

    Tolerances apply separately to every entry, including weak mutual terms.
    Charge capacitances are checked with the readout both grounded and floating.
    ``atol`` is in F. An allowance fraction at most one passes the check.
    These checks run eagerly, outside JAX tracing.

    Returns:
        JSON-compatible errors and a combined pass flag.

    Raises:
        ValueError: If terminal conventions, tolerances or matrices are invalid.
    """
    a, b = jnp.asarray(actual), jnp.asarray(reference)
    n = len(terminals)
    if a.shape != (n, n) or b.shape != (n, n):
        raise ValueError("Matrices must match the ordered terminals")
    if (
        len(set(terminals)) != n
        or junction[0] not in terminals
        or junction[1] == junction[0]
        or (junction[1] is not None and junction[1] not in terminals)
    ):
        raise ValueError("Junction must name distinct terminals or reference ground")
    if rtol < 0 or atol <= 0:
        raise ValueError("Require nonnegative rtol and positive absolute tolerance")
    check_maxwell(a)
    check_maxwell(b)
    positive = terminals.index(junction[0])
    negative = None if junction[1] is None else terminals.index(junction[1])

    def charge(matrix: jnp.ndarray, *, grounded: bool) -> jnp.ndarray:
        if negative is None:
            return (
                matrix[positive, positive]
                if grounded
                else 1 / jnp.linalg.inv(matrix)[positive, positive]
            )
        if grounded:
            indices = jnp.asarray([positive, negative])
            return differential_capacitance(matrix[indices[:, None], indices[None, :]])
        return differential_capacitance(matrix, positive, negative)

    comparisons = {
        "matrix": (a, b),
        "ground": (a.sum(-1), b.sum(-1)),
        "qubit_readout_grounded": (charge(a, grounded=True), charge(b, grounded=True)),
        "qubit_readout_floating": (
            charge(a, grounded=False),
            charge(b, grounded=False),
        ),
    }
    errors = {
        name: {
            "max_absolute_F": float(jnp.max(jnp.abs(x - y))),
            "allowance_fraction": float(
                jnp.max(jnp.abs(x - y) / (atol + rtol * jnp.abs(y)))
            ),
        }
        for name, (x, y) in comparisons.items()
    }
    return {
        "errors": errors,
        "passed": all(error["allowance_fraction"] <= 1 for error in errors.values()),
    }


def validate_native_capacitances(
    frame: "pl.DataFrame",
    metadata: "DatasetMetadata",
    grid: Mapping[str, Sequence[float]],
    variants: Mapping[str, str],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Check eager native capacitance tables against an externally frozen grid.

    Every declared quantity must cover the full task grid before gridding can
    infer a smaller domain. Maxwell and mutual matrices must agree. The FEM
    indicator is nonnegative, not an accuracy bound. Iterations are nonnegative
    integers and relative residuals lie between zero and the solver tolerance.
    Synthetic fixtures can exercise these checks without claiming FEM admission.

    Returns:
        Raw values in metadata axis and terminal order for embedding as evidence.

    Raises:
        ValueError: If quantities, coordinates, matrices or diagnostics are invalid.
    """
    axes = [axis.name for axis in metadata.axes]
    if set(grid) != set(axes) or any(
        not len(coords)
        or list(coords) != sorted(set(coords))
        or not all(isfinite(value) for value in coords)
        for coords in grid.values()
    ):
        raise ValueError(
            "Native grid must have finite, unique, ascending metadata axes"
        )
    required = {
        "maxwell_capacitance",
        "mutual_capacitance",
        "fem_error_indicator_norm",
        "solver_iterations",
        "solver_relative_residual",
    }
    if not required <= {q.name for q in metadata.quantities}:
        raise ValueError("Native capacitance quantities or diagnostics are missing")
    for quantity in metadata.quantities:
        if quantity.name not in required:
            continue
        kind = {
            "maxwell_capacitance": QuantityKind.MAXWELL_CAPACITANCE,
            "mutual_capacitance": QuantityKind.MUTUAL_CAPACITANCE,
        }.get(quantity.name, QuantityKind.CIRCUIT_PARAMETER)
        if (
            quantity.kind != kind
            or quantity.complex
            or quantity.unit != ("1" if kind == QuantityKind.CIRCUIT_PARAMETER else "F")
        ):
            raise ValueError("Native capacitance quantity kinds or units differ")
    if set(variants) != set(metadata.variants):
        raise ValueError("Native variants differ from metadata")
    validate(frame, metadata)
    if any(
        frame[name].unique().to_list() != [value] for name, value in variants.items()
    ):
        raise ValueError("Native variant differs from its task")
    values = {}
    for quantity in metadata.quantities:
        selected = frame.filter(pl.col("quantity") == quantity.name)
        if any(
            selected[name].unique().sort().to_list() != list(grid[name])
            for name in axes
        ):
            raise ValueError(
                f"Native quantity {quantity.name} differs from its task grid"
            )
        values[quantity.name] = to_grid(
            frame, metadata, quantity.name, allow_missing=False, **variants
        ).values
    matrix = values["maxwell_capacitance"]
    check_maxwell(matrix)
    if not bool(
        jnp.allclose(
            values["mutual_capacitance"], maxwell_to_mutual(matrix), rtol=1e-9, atol=0
        )
    ):
        raise ValueError("Native Maxwell and mutual capacitances disagree")
    for name in (
        "fem_error_indicator_norm",
        "solver_iterations",
        "solver_relative_residual",
    ):
        samples = jnp.asarray(values[name])
        if not bool(jnp.isfinite(samples).all() & (samples >= 0).all()):
            raise ValueError(f"Native {name} must be finite and nonnegative")
    iterations = jnp.asarray(values["solver_iterations"])
    if not bool((iterations == jnp.floor(iterations)).all()):
        raise ValueError("Native solver_iterations must be integral")
    if (
        not isfinite(tolerance)
        or tolerance < 0
        or bool((jnp.asarray(values["solver_relative_residual"]) > tolerance).any())
    ):
        raise ValueError("Native residual exceeds its tolerance")
    return {name: array.tolist() for name, array in values.items()}
