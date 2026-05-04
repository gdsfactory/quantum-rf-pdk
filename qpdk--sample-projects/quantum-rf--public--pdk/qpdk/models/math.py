"""Mathematical helper functions for models."""

from functools import partial

import jax
import jaxellip
from jax.typing import ArrayLike

from qpdk.models.constants import ε_0


@partial(jax.jit, inline=True)
def ellipk_ratio(m: ArrayLike) -> jax.Array:
    """Ratio of complete elliptic integrals of the first kind K(m) / K(1-m)."""
    return jaxellip.ellipk(m) / jaxellip.ellipk(1 - m)


@partial(jax.jit, inline=True)
def epsilon_eff(ep_r: ArrayLike) -> jax.Array:
    """Effective permittivity for a substrate with relative permittivity ep_r."""
    return (ep_r + 1) / 2


@partial(jax.jit, inline=True)
def capacitance_per_length_conformal(
    m: ArrayLike,
    ep_r: ArrayLike,
) -> jax.Array:
    """Calculate capacitance per unit length using conformal mapping.

    C_pul = ε_0 * ε_eff * K(m)/K(1-m)
    ε_eff = (ep_r + 1) / 2.

    Args:
        m: The parameter m (modulus squared) of the elliptic integral.
        ep_r: Relative permittivity of the substrate.

    Returns:
        The capacitance per unit length in Farads/meter.
    """
    return ε_0 * epsilon_eff(ep_r) * ellipk_ratio(m)
