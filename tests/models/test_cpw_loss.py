"""Tests for CPW dielectric loss functions."""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

import qpdk
import qpdk.models.cpw as cpw_mod
from qpdk.models.waveguides import straight


@pytest.fixture(autouse=True)
def activate_pdk():
    """Ensure PDK is active for all tests."""
    qpdk.PDK.activate()


class TestCPWLossParameters:
    """Tests for the complex effective permittivity calculation."""

    @staticmethod
    def test_lossless_is_real() -> None:
        """With tand=0.0, ep_eff should be purely real."""
        ep_eff, _z0 = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=0.0)
        assert jnp.isrealobj(ep_eff) or jnp.all(jnp.imag(ep_eff) == 0)

    @staticmethod
    def test_lossy_is_complex() -> None:
        """With tand > 0, ep_eff should have a negative imaginary part."""
        # Realistic loss tangent
        tand = 1e-4
        ep_eff, _z0 = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=tand)

        assert jnp.imag(ep_eff) < 0, (
            f"Imaginary part should be negative (got {jnp.imag(ep_eff)})"
        )
        # Check magnitude is reasonable: epsilon'' ~ epsilon' * tand * q
        # For CPW, q ~ 0.5, so epsilon'' ~ 6 * 1e-4 * 0.5 ~ 3e-4
        assert jnp.abs(jnp.imag(ep_eff)) > 1e-6

    @staticmethod
    def test_loss_scaling() -> None:
        """The imaginary part of ep_eff should scale linearly with tand."""
        ep_eff_1, _ = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=1e-4)
        ep_eff_2, _ = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=2e-4)

        ratio = jnp.imag(ep_eff_2) / jnp.imag(ep_eff_1)
        assert_allclose(float(ratio), 2.0, rtol=1e-5)

    @staticmethod
    def test_default_pdk_loss() -> None:
        """Verify that omitting tand uses the PDK default (set to 2.7e-6)."""
        _h, _t, _ep_r, tand_pdk = cpw_mod.get_cpw_substrate_params()
        # The default value should be 2.7e-6 in the technology definition
        assert tand_pdk == pytest.approx(2.7e-6)

        ep_eff_default, _ = cpw_mod.cpw_parameters(width=10.0, gap=6.0)
        ep_eff_explicit, _ = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=2.7e-6)

        assert_allclose(ep_eff_default, ep_eff_explicit)
        if tand_pdk > 0:
            assert jnp.imag(ep_eff_default) < 0


class TestStraightAttenuation:
    """Tests for attenuation in straight CPW models."""

    @staticmethod
    def test_lossy_straight_attenuation(monkeypatch) -> None:
        """A lossy straight CPW should have |S21| < 1."""
        f = jnp.linspace(1e9, 10e9, 11)

        # Use monkeypatch to avoid leaking state
        original_params = cpw_mod.cpw_parameters

        def mock_params(w, g, tand=None):
            ep, z0 = original_params(w, g, tand)
            return ep.real * (1 - 0.01j), z0

        monkeypatch.setattr(cpw_mod, "cpw_parameters", mock_params)

        s = straight(f=f, length=10000.0)  # 10mm
        s21 = jnp.abs(s["o2", "o1"])
        assert jnp.all(s21 < 1.0)
        assert jnp.all(s21 > 0.5)  # Should not be completely dead

    @staticmethod
    def test_attenuation_scales_with_length(monkeypatch) -> None:
        """Longer lossy lines should have more attenuation."""
        f = jnp.array([5e9])

        # Use monkeypatch to avoid leaking state
        original_params = cpw_mod.cpw_parameters

        def mock_params(w, g, tand=None):
            ep, z0 = original_params(w, g, tand)
            return ep.real * (1 - 0.01j), z0

        monkeypatch.setattr(cpw_mod, "cpw_parameters", mock_params)

        s_short = straight(f=f, length=1000.0)
        s_long = straight(f=f, length=10000.0)

        assert jnp.abs(s_long["o2", "o1"]) < jnp.abs(s_short["o2", "o1"])


class TestNumericalStability:
    """Tests for edge cases and numerical stability."""

    @staticmethod
    def test_vacuum_loss_stability(monkeypatch) -> None:
        """Verify that loss correction does not crash for ep_r ~ 1."""
        # Clear cache to ensure mock is used
        cpw_mod.cpw_parameters.cache_clear()
        cpw_mod.get_cpw_substrate_params.cache_clear()

        # Mock get_cpw_substrate_params to return ep_r = 1.0
        monkeypatch.setattr(
            cpw_mod, "get_cpw_substrate_params", lambda: (500.0, 0.2, 1.0, 1e-4)
        )

        # This shouldn't raise ZeroDivisionError now
        ep_eff, _z0 = cpw_mod.cpw_parameters(width=10.0, gap=6.0, tand=1e-4)
        assert jnp.isfinite(ep_eff).all()
        # For ep_r = 1, ep_eff should be real (no dielectric loss contribution from substrate)
        assert jnp.all(jnp.imag(ep_eff) == 0)
