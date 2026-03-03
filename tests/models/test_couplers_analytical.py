"""Tests for analytical ECCPW mutual capacitance formula."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.couplers import cpw_cpw_coupling_capacitance_analytical


def test_cpw_cpw_coupling_capacitance_positivity():
    """Verify the formula returns a positive value for all valid CPW geometries."""
    # Typical parameters
    length = 20.0
    gap = 0.27
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    c_m = cpw_cpw_coupling_capacitance_analytical(
        length=length, gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )
    assert c_m > 0


def test_cpw_cpw_coupling_capacitance_monotonicity():
    """Verify coupling capacitance decreases monotonically as inter-CPW gap increases."""
    length = 20.0
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    gaps = np.linspace(0.1, 10.0, 10)
    capacitances = [
        float(
            cpw_cpw_coupling_capacitance_analytical(
                length=length, gap=g, width=width, cpw_gap=cpw_gap, ep_r=ep_r
            )
        )
        for g in gaps
    ]

    # Check if strictly decreasing
    assert all(x > y for x, y in itertools.pairwise(capacitances))


def test_cpw_cpw_coupling_capacitance_consistency():
    """Check consistency with expected values for a known geometry.

    For gap=0.27 µm, width=10 µm, cpw_gap=6 µm, ep_r=11.7, length=20 µm,
    the expected coupling capacitance should be in the range of 1-5 fF.
    """
    length = 20.0
    gap = 0.27
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    c_m = cpw_cpw_coupling_capacitance_analytical(
        length=length, gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Convert to femtoFarads
    c_m_ff = float(c_m) * 1e15

    # Check that it's in a physically reasonable range for these dimensions
    # Based on similar CPW coupler designs in literature
    assert 1.0 < c_m_ff < 5.0

    # Test with different length (should scale linearly)
    c_m_long = cpw_cpw_coupling_capacitance_analytical(
        length=2 * length, gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )
    assert jnp.isclose(c_m_long, 2 * c_m)


@pytest.mark.parametrize("ep_r", [1.0, 4.4, 11.7])
def test_cpw_cpw_coupling_capacitance_permittivity_scaling(ep_r):
    """Verify that capacitance scales with effective permittivity (ep_r + 1)."""
    length = 20.0
    gap = 0.5
    width = 10.0
    cpw_gap = 6.0

    c_m = cpw_cpw_coupling_capacitance_analytical(
        length=length, gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Reference with vacuum (ep_r = 1.0)
    c_m_vacuum = cpw_cpw_coupling_capacitance_analytical(
        length=length, gap=gap, width=width, cpw_gap=cpw_gap, ep_r=1.0
    )

    # ep_eff = (ep_r + 1) / 2
    # c_m should be proportional to ep_eff
    ep_eff_ratio = ((ep_r + 1) / 2) / ((1.0 + 1) / 2)
    assert jnp.isclose(c_m, c_m_vacuum * ep_eff_ratio)
