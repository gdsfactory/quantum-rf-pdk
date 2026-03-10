"""Additional tests for qpdk.models.couplers module - covering missing lines."""

from typing import final

import jax.numpy as jnp
import numpy as np

from qpdk.models.couplers import (
    coupler_straight,
    cpw_cpw_coupling_capacitance,
    cpw_cpw_coupling_capacitance_per_length_analytical,
)

from .base import FourPortModelTestSuite


@final
class TestCouplerStraight(FourPortModelTestSuite):
    """Tests for coupler_straight model."""

    model_function = coupler_straight


class TestCPWCouplingCapacitanceAnalytical:
    """Tests for cpw_cpw_coupling_capacitance_per_length_analytical."""

    @staticmethod
    def test_positive_capacitance() -> None:
        """Test that coupling capacitance per length is positive."""
        c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=5.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert float(c_pul) > 0

    @staticmethod
    def test_capacitance_decreases_with_gap() -> None:
        """Test that capacitance decreases as gap increases."""
        c_small_gap = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=1.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        c_large_gap = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=10.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert float(c_small_gap) > float(c_large_gap)

    @staticmethod
    def test_broadcasting() -> None:
        """Test that broadcasting works for multiple gaps."""
        gaps = jnp.geomspace(0.5, 5.0, 5)
        c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=gaps, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert c_pul.shape == (5,)
        assert jnp.all(c_pul > 0)


class TestCPWCouplingCapacitance:
    """Tests for cpw_cpw_coupling_capacitance."""

    @staticmethod
    def test_total_capacitance_scales_with_length() -> None:
        """Test that total coupling capacitance scales linearly with length."""
        f = jnp.array([5e9])
        c1 = cpw_cpw_coupling_capacitance(f, length=100.0, gap=5.0, cross_section="cpw")
        c2 = cpw_cpw_coupling_capacitance(f, length=200.0, gap=5.0, cross_section="cpw")
        np.testing.assert_allclose(float(c2), 2.0 * float(c1), rtol=1e-6)
