"""Tests for coupler model fallback path when cross-section lacks CPW sections."""

import gdsfactory as gf
import jax.numpy as jnp

from qpdk.models.couplers import cpw_cpw_coupling_capacitance, coupler_straight


class TestCouplerCpwFallback:
    """Test the fallback path in cpw_cpw_coupling_capacitance."""

    @staticmethod
    def test_fallback_with_plain_cross_section() -> None:
        """Test that a cross-section without etch_offset sections triggers fallback."""
        # Create a plain cross-section without CPW gap sections
        xs = gf.cross_section.cross_section(width=10.0)
        f = jnp.linspace(1e9, 10e9, 5)

        # Should not raise — should use fallback with default gap=6.0
        result = cpw_cpw_coupling_capacitance(f=f, length=100.0, gap=1.0, cross_section=xs)
        assert result is not None
        # Result should be positive (physical capacitance)
        assert float(result) > 0

    @staticmethod
    def test_coupler_straight_with_default_cross_section() -> None:
        """Test coupler_straight works with the default cross-section."""
        f = jnp.linspace(1e9, 10e9, 10)
        result = coupler_straight(f=f, length=100.0, gap=1.0)
        assert isinstance(result, dict)
        # Should have 4-port S-parameters
        assert ("o1", "o1") in result
