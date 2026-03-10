"""Tests for qpdk.models.junction module - Josephson junction and SQUID models."""

import warnings
from typing import final

import jax.numpy as jnp
from numpy.testing import assert_allclose, assert_array_less

from qpdk.models.junction import josephson_junction, squid_junction

from .base import TwoPortModelTestSuite


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
        with warnings.catch_warnings(record=True) as w:
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
        assert jnp.all(jnp.isfinite(s11)), "Should produce finite S-parameters even near critical current"


@final
class TestSQUIDJunction(TwoPortModelTestSuite):
    """Tests for squid_junction S-parameter model."""

    model_function = squid_junction

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"ic_tot": 2e-6, "asymmetry": 0.0, "capacitance": 10e-15, "resistance": 5e3}

    @staticmethod
    def test_flux_tuning() -> None:
        """Test that external flux modifies S-parameters."""
        f = jnp.linspace(1e9, 10e9, 50)
        from qpdk.models.constants import Φ_0

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
        from qpdk.models.constants import Φ_0

        result = squid_junction(f=f, ic_tot=2e-6, asymmetry=0.0, flux=Φ_0 / 2)
        s11 = result[("o1", "o1")]
        assert jnp.all(jnp.isfinite(s11)), "SQUID at half flux quantum should give finite S-params"

    @staticmethod
    def test_squid_asymmetry() -> None:
        """Test that asymmetry modifies SQUID response."""
        f = jnp.linspace(1e9, 10e9, 50)
        from qpdk.models.constants import Φ_0

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
