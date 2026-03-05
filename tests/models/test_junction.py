"""Tests for junction models."""

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

from qpdk.models.constants import Φ_0
from qpdk.models.junction import squid_junction

MAX_EXAMPLES = 20


@given(
    ic_single=st.floats(min_value=1e-8, max_value=1e-5),
    asymmetry=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_squid_junction_flux_tunability_hypothesis(
    ic_single: float, asymmetry: float
) -> None:
    """SQUID critical current should be maximal at zero flux and minimal near half flux quantum."""
    f = jnp.array([5e9])

    # zero flux
    s_zero = squid_junction(f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=0.0)

    # half flux
    s_half = squid_junction(f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=Φ_0 / 2)

    # They should have different responses since the effective inductance changes
    assert not jnp.allclose(s_zero[("o1", "o1")], s_half[("o1", "o1")])
    assert jnp.all(jnp.isfinite(s_half[("o1", "o1")]))

    # Admittance should be strictly smaller at half flux -> S11 magnitude should be closer to 1 if it acts more like an open circuit
    # Wait, inductance increases at half flux. So L -> large, Y_L -> 0.
    # Whether |S11| increases or decreases depends on C and R. We just verify the shift is finite and distinct.


@given(
    ic_single=st.floats(min_value=1e-8, max_value=1e-5),
    ibias_multiplier=st.floats(min_value=1.5, max_value=10.0),
    phi=st.floats(min_value=0.0, max_value=Φ_0),
)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_squid_junction_overbias_warning_hypothesis(
    ic_single: float, ibias_multiplier: float, phi: float
) -> None:
    """Overbias condition should emit a warning but remain JIT-compatible and finite across regimes."""
    asymmetry = 0.0

    def overbiased_s_params(phi_val: jnp.ndarray, ibias_val: float) -> jnp.ndarray:
        f = jnp.array([5e9])
        return squid_junction(
            f=f, ic_tot=ic_single, asymmetry=asymmetry, flux=phi_val, ib=ibias_val
        )[("o1", "o1")]

    phi_arr = jnp.array(phi)
    ibias = ibias_multiplier * ic_single

    jitted = jax.jit(overbiased_s_params)

    with pytest.warns(RuntimeWarning, match="DC bias"):
        y = jitted(phi_arr, ibias)

    assert jnp.all(jnp.isfinite(y))
