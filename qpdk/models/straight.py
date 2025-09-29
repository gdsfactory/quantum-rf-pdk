"""S-parameter model for a straight waveguide."""

import jax.numpy as jnp
import sax
from jax.typing import ArrayLike


def straight(f: ArrayLike = 5e9) -> sax.SType:
    """S-parameter model for a straight waveguide.

    Args:
        f: Frequency in Hz

    Returns:
        sax.SType: S-parameters dictionary
    """
    f = jnp.asarray(f)
    sdict = {("o1", "o2"): jnp.ones_like(f)}
    return sax.reciprocal(sdict)
