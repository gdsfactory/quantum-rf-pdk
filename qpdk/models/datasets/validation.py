"""Independent accuracy checks for electrostatic qubit datasets."""

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike

from qpdk.models.datasets.capacitance import check_maxwell, differential_capacitance


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
