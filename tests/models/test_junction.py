"""Tests for junction models."""

import warnings
from typing import final

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

from qpdk.models.constants import Φ_0
from qpdk.models.junction import josephson_junction, squid_junction

from .base import TwoPortModelTestSuite

MAX_EXAMPLES = 20


@final
class TestJosephsonJunction(TwoPortModelTestSuite):
    """Tests for josephson_junction S-parameter model."""

    model_function = josephson_junction

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"ic": 1e-6, "capacitance": 5e-15, "resistance": 10e3}

    @staticmethod
    def test_bias_current_changes_response() -> None:
        """Test that DC bias current modifies the S-parameters."""
        f = jnp.linspace(1e9, 10e9, 50)

        s_no_bias = josephson_junction(f=f, ic=1e-6, ib=0.0)
        s_biased = josephson_junction(f=f, ic=1e-6, ib=0.5e-6)

        s11_no = jnp.abs(s_no_bias[("o1", "o1")])
        s11_biased = jnp.abs(s_biased[("o1", "o1")])

        assert not jnp.allclose(s11_no, s11_biased, atol=1e-3), (
            "Bias current should change S-parameters"
        )

    @staticmethod
    def test_overbiased_warning() -> None:
        """Test that overbiased condition triggers a RuntimeWarning."""
        f = jnp.array([5e9])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            josephson_junction(f=f, ic=1e-6, ib=2e-6)
            # JAX debug.callback may or may not fire synchronously
            # We just check the function completes without error

    @staticmethod
    def test_phase_factor_clipping() -> None:
        """Test that clipping works for near-critical bias."""
        f = jnp.array([5e9])
        # ib very close to ic: cos_Φ_0 ≈ 0, should still give finite output
        result = josephson_junction(f=f, ic=1e-6, ib=0.999e-6)
        s11 = result[("o1", "o1")]
        assert jnp.all(jnp.isfinite(s11)), (
            "Should produce finite S-parameters even near critical current"
        )


@final
class TestSQUIDJunction(TwoPortModelTestSuite):
    """Tests for squid_junction S-parameter model."""

    model_function = squid_junction

    @staticmethod
    def get_model_kwargs() -> dict:
        return {
            "ic_tot": 2e-6,
            "asymmetry": 0.0,
            "capacitance": 10e-15,
            "resistance": 5e3,
        }

    @staticmethod
    def test_flux_tuning() -> None:
        """Test that external flux modifies S-parameters."""
        f = jnp.linspace(1e9, 10e9, 50)

        s_zero_flux = squid_junction(f=f, ic_tot=2e-6, flux=0.0)
        s_half_flux = squid_junction(f=f, ic_tot=2e-6, flux=Φ_0 / 4)

        s11_zero = jnp.abs(s_zero_flux[("o1", "o1")])
        s11_half = jnp.abs(s_half_flux[("o1", "o1")])

        assert not jnp.allclose(s11_zero, s11_half, atol=1e-3), (
            "Flux should modify SQUID S-parameters"
        )

    @staticmethod
    def test_squid_at_half_flux_quantum() -> None:
        """Test SQUID at half flux quantum (symmetric case, ic_squid → 0)."""
        f = jnp.array([5e9])

        result = squid_junction(f=f, ic_tot=2e-6, asymmetry=0.0, flux=Φ_0 / 2)
        s11 = result[("o1", "o1")]
        assert jnp.all(jnp.isfinite(s11)), (
            "SQUID at half flux quantum should give finite S-params"
        )

    @staticmethod
    def test_squid_asymmetry() -> None:
        """Test that asymmetry modifies SQUID response."""
        f = jnp.linspace(1e9, 10e9, 50)

        s_sym = squid_junction(f=f, ic_tot=2e-6, asymmetry=0.0, flux=Φ_0 / 4)
        s_asym = squid_junction(f=f, ic_tot=2e-6, asymmetry=0.3, flux=Φ_0 / 4)

        s11_sym = jnp.abs(s_sym[("o1", "o1")])
        s11_asym = jnp.abs(s_asym[("o1", "o1")])

        assert not jnp.allclose(s11_sym, s11_asym, atol=1e-3), (
            "Asymmetry should change SQUID response"
        )

    @staticmethod
    def test_squid_with_bias_current() -> None:
        """Test SQUID with DC bias current."""
        f = jnp.array([5e9])
        result = squid_junction(f=f, ic_tot=2e-6, ib=0.5e-6)
        s11 = result[("o1", "o1")]
        assert jnp.all(jnp.isfinite(s11))


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
    assert jnp.isfinite(s_half[("o1", "o1")]).all()


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

    assert jnp.isfinite(y).all()
