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


class NonPhysicalMatrixError(ValueError):
    """A Maxwell capacitance matrix violates a physical property."""

    def __init__(self, problems: list[str]) -> None:
        """Store the violated properties and list them in the message."""
        self.problems = problems
        super().__init__(f"Non-physical Maxwell matrix: {'; '.join(problems)}.")


def check_maxwell(maxwell: ArrayLike, *, rtol: float = 1e-9) -> None:
    """Check that a Maxwell capacitance matrix is physical.

    Checks symmetry, positive diagonal, non-positive off-diagonal entries, and
    diagonal dominance (non-negative capacitance to ground), each up to ``rtol``
    relative to the largest diagonal entry of each matrix. Non-finite entries are reported
    before any other check, since they make the tolerance meaningless. Runs
    eagerly on concrete arrays; it is a check, not a model, and is not jittable.

    Raises:
        NonPhysicalMatrixError: Listing every violated property.
    """
    c = jnp.asarray(maxwell, dtype=float)
    if not jnp.isfinite(c).all():
        raise NonPhysicalMatrixError(["contains non-finite entries"])
    diagonal = jnp.diagonal(c, axis1=-2, axis2=-1)
    off = ~jnp.eye(c.shape[-1], dtype=bool)
    # Per-matrix tolerance, so a large matrix in a batch cannot hide a small one.
    tol = rtol * jnp.abs(diagonal).max(axis=-1, keepdims=True)
    checks = {
        "not symmetric": (jnp.abs(c - jnp.swapaxes(c, -1, -2)) > tol[..., None]).any(),
        "non-positive diagonal": (diagonal <= 0).any(),
        "positive off-diagonal": (jnp.where(off, c, 0) > tol[..., None]).any(),
        "negative capacitance to ground": (c.sum(axis=-1) < -tol).any(),
    }
    if problems := [problem for problem, violated in checks.items() if violated]:
        raise NonPhysicalMatrixError(problems)


def differential_capacitance(
    maxwell: ArrayLike, positive: int = 0, negative: int = 1
) -> jnp.ndarray:
    r"""Charging capacitance for equal and opposite island charges.

    All other named conductors float with zero incremental charge. The reference
    ground remains at zero voltage. For :math:`d = e_p - e_n`, this returns
    :math:`(d^{\text{T}} C^{-1} d)^{-1}`. Grounding another conductor instead requires
    deleting its row and column before this reduction. Junction capacitance
    is a separate parallel contribution.

    Returns:
        Differential capacitance in F, preserving leading batch dimensions.
    """
    c = jnp.asarray(maxwell)
    d = (
        jnp.eye(c.shape[-1], dtype=c.dtype)[positive]
        - jnp.eye(c.shape[-1], dtype=c.dtype)[negative]
    )
    potentials = jnp.linalg.solve(
        c, jnp.broadcast_to(d, (*c.shape[:-2], c.shape[-1]))[..., None]
    )[..., 0]
    return 1 / jnp.sum(d * potentials, axis=-1)
