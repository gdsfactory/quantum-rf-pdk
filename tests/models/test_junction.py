"""Tests for junction models."""

import jax
import jax.numpy as jnp
import pytest

from qpdk.models.constants import Φ_0
from qpdk.models.junction import squid_junction


def test_squid_junction_flux_tunability() -> None:
    """SQUID critical current should be maximal at zero flux and minimal near half flux quantum."""
    ic_single = 1.0e-6  # A
    asymmetry = 0.0

    # Test the S-parameters directly.
    # At half-flux, ic -> 0, L_J -> inf, Y_L -> 0.
    f = jnp.array([5e9])

    # zero flux
    s_zero = squid_junction(f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=0.0)

    # half flux
    s_half = squid_junction(f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=Φ_0 / 2)

    # Should not be equal, and half flux should be finite
    assert not jnp.allclose(s_zero[("o1", "o1")], s_half[("o1", "o1")])
    assert jnp.all(jnp.isfinite(s_half[("o1", "o1")]))


def test_squid_junction_overbias_warning_jittable() -> None:
    """Overbias condition should emit a warning but remain JIT-compatible."""
    ic_single = 1.0e-6
    asymmetry = 0.0

    def overbiased_s_params(phi: jnp.ndarray, ibias: float) -> jnp.ndarray:
        f = jnp.array([5e9])
        # Returns S11
        return squid_junction(
            f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=phi, ib=ibias
        )[("o1", "o1")]

    phi = jnp.array(0.0)
    ibias = 5.0 * ic_single  # well above critical current to trigger overbias

    # JIT compilation should succeed even if a warning is emitted at runtime.
    jitted = jax.jit(overbiased_s_params)

    # Ensure call works and returns a finite admittance (no NaNs/Infs).
    with pytest.warns(RuntimeWarning, match="DC bias"):
        y = jitted(phi, ibias)

    assert jnp.all(jnp.isfinite(y))
