"""Conversions and physical checks for capacitance matrices.

Matrices have terminals on the last two axes, so leading batch axes from
interpolated lookups pass straight through.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike


def maxwell_to_mutual(maxwell: ArrayLike) -> jnp.ndarray:
    r"""Convert a Maxwell capacitance matrix to the mutual (SPICE-style) matrix.

    Off-diagonal entries become the branch capacitances :math:`C_{ij}^m = -C_{ij}`,
    and diagonal entries become the capacitance of each terminal to the reference
    ground, :math:`C_{ii}^m = \sum_j C_{ij}`.

    Returns:
        Mutual capacitance matrix, same shape as ``maxwell``.
    """
    c = jnp.asarray(maxwell)
    n = c.shape[-1]
    eye = jnp.eye(n, dtype=bool)
    return jnp.where(eye, c.sum(axis=-1)[..., None] * jnp.ones(n), -c)


def maxwell_violations(maxwell: ArrayLike, *, rtol: float = 1e-9) -> list[str]:
    """List the physical properties a Maxwell capacitance matrix violates.

    Checks symmetry, positive diagonal, non-positive off-diagonal entries, and
    diagonal dominance (non-negative capacitance to ground), each up to ``rtol``
    relative to the largest diagonal entry. Non-finite entries are reported
    before any other check, since they make the tolerance meaningless. Runs
    eagerly on concrete arrays; it is a check, not a model, and is not jittable.

    Returns:
        Descriptions of the violated properties; empty if the matrix is physical.
    """
    c = jnp.asarray(maxwell, dtype=float)
    if not jnp.isfinite(c).all():
        return ["contains non-finite entries"]
    diagonal = jnp.diagonal(c, axis1=-2, axis2=-1)
    off = ~jnp.eye(c.shape[-1], dtype=bool)
    tol = rtol * jnp.abs(diagonal).max()
    checks = {
        "not symmetric": jnp.abs(c - jnp.swapaxes(c, -1, -2)).max() > tol,
        "non-positive diagonal": (diagonal <= 0).any(),
        "positive off-diagonal": jnp.where(off, c, 0).max() > tol,
        "negative capacitance to ground": (c.sum(axis=-1) < -tol).any(),
    }
    return [problem for problem, violated in checks.items() if violated]
