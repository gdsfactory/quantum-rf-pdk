"""Conversions and physical checks for capacitance matrices.

Matrices have terminals on the last two axes, so leading batch axes from
interpolated lookups pass straight through.
"""

import jax.numpy as jnp
import numpy as np
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
    relative to the largest diagonal entry. NaN entries are reported.

    Returns:
        Descriptions of the violated properties; empty if the matrix is physical.
    """
    c = np.asarray(maxwell, dtype=float)
    if np.isnan(c).any():
        return ["contains NaN"]
    n = c.shape[-1]
    off = ~np.eye(n, dtype=bool)
    tol = rtol * np.abs(np.diagonal(c, axis1=-2, axis2=-1)).max()
    problems = []
    if np.abs(c - np.swapaxes(c, -1, -2)).max() > tol:
        problems.append("not symmetric")
    if (np.diagonal(c, axis1=-2, axis2=-1) <= 0).any():
        problems.append("non-positive diagonal")
    if (c[..., off] > tol).any():
        problems.append("positive off-diagonal")
    if (c.sum(axis=-1) < -tol).any():
        problems.append("negative capacitance to ground")
    return problems
