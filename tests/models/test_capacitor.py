"""Tests for qpdk.models.capacitor module."""

import jax.numpy as jnp
import numpy as np

from qpdk.models.capacitor import interdigital_capacitor, plate_capacitor


def test_plate_capacitor() -> None:
    """Test plate_capacitor model."""
    f = jnp.linspace(1e9, 10e9, 11)
    result = plate_capacitor(f=f, length=26.0, width=5.0, gap=7.0)

    assert isinstance(result, dict)
    assert ("o1", "o1") in result
    assert ("o1", "o2") in result
    assert len(result[("o1", "o1")]) == len(f)


def test_interdigital_capacitor() -> None:
    """Test interdigital_capacitor model."""
    f = jnp.linspace(1e9, 10e9, 11)

    # Test with N=2
    result_n2 = interdigital_capacitor(f=f, fingers=2)
    assert isinstance(result_n2, dict)

    # Test with N=4 (default)
    result_n4 = interdigital_capacitor(f=f, fingers=4)
    assert isinstance(result_n4, dict)

    # Capacitance should increase with number of fingers
    # Note: we are comparing S-parameters, but for a simple capacitor model
    # more capacitance means lower impedance, so more transmission (|S21|)
    # and less reflection (|S11|) at the same frequency.
    s11_n2 = jnp.abs(result_n2[("o1", "o1")])
    s11_n4 = jnp.abs(result_n4[("o1", "o1")])

    # At 5GHz, N=4 should have more capacitance than N=2
    assert jnp.all(s11_n4 <= s11_n2 + 1e-10)


def test_interdigital_capacitor_scaling() -> None:
    """Test that interdigital_capacitor capacitance scales correctly."""
    from qpdk.models.capacitor import _get_interdigital_capacitor_extraction_results

    c2 = _get_interdigital_capacitor_extraction_results(fingers=2)
    c4 = _get_interdigital_capacitor_extraction_results(fingers=4)
    c6 = _get_interdigital_capacitor_extraction_results(fingers=6)

    assert c2 > 0
    assert c4 > c2
    assert c6 > c4

    # Scaling should be roughly linear for large N
    # C(N) = (N-3)CI/2 + const
    # C(6) - C(4) = (3CI/2 + const) - (CI/2 + const) = CI
    # C(8) - C(6) = CI
    c8 = _get_interdigital_capacitor_extraction_results(fingers=8)
    diff1 = c6 - c4
    diff2 = c8 - c6
    assert np.isclose(diff1, diff2, rtol=1e-10)
