"""Tests for qpdk.models.capacitor module."""

from typing import TYPE_CHECKING, final

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_less

from qpdk.models.capacitor import (
    interdigital_capacitor,
    plate_capacitor,
    plate_capacitor_capacitance_analytical,
)

from .base import TwoPortModelTestSuite

if TYPE_CHECKING:
    pass


@final
class TestPlateCapacitor(TwoPortModelTestSuite):
    """Test plate_capacitor model."""

    model_function = staticmethod(plate_capacitor)

    @staticmethod
    def get_model_kwargs() -> dict:
        return {"length": 26.0, "width": 5.0, "gap": 7.0}


@final
class TestInterdigitalCapacitor(TwoPortModelTestSuite):
    """Test interdigital_capacitor model."""

    model_function = staticmethod(interdigital_capacitor)

    def test_scaling_with_fingers(self) -> None:
        """Test that capacitance increases with number of fingers."""
        f = self.get_frequency_array(11)

        # Test with N=2
        result_n2 = self._call_model(f=f, fingers=2)

        # Test with N=4 (default)
        result_n4 = self._call_model(f=f, fingers=4)

        # Capacitance should increase with number of fingers
        # Note: we are comparing S-parameters, but for a simple capacitor model
        # more capacitance means lower impedance, so more transmission (|S21|)
        # and less reflection (|S11|) at the same frequency.
        s11_n2 = jnp.abs(result_n2[("o1", "o1")])
        s11_n4 = jnp.abs(result_n4[("o1", "o1")])

        # At 5GHz, N=4 should have more capacitance than N=2
        assert_array_less(s11_n4, s11_n2 + 1e-10)


def test_interdigital_capacitor_scaling() -> None:
    """Test that interdigital_capacitor capacitance scales correctly."""
    from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical

    kwargs = {
        "finger_length": 20.0,
        "finger_gap": 2.0,
        "thickness": 5.0,
        "ep_r": 10.0,
    }

    c2 = interdigital_capacitor_capacitance_analytical(fingers=2, **kwargs)
    c4 = interdigital_capacitor_capacitance_analytical(fingers=4, **kwargs)
    c6 = interdigital_capacitor_capacitance_analytical(fingers=6, **kwargs)

    assert c2 > 0
    assert c4 > c2
    assert c6 > c4

    # Scaling should be roughly linear for large N
    # C(N) = (N-3)CI/2 + const
    # C(6) - C(4) = (3CI/2 + const) - (CI/2 + const) = CI
    # C(8) - C(6) = CI
    c8 = interdigital_capacitor_capacitance_analytical(fingers=8, **kwargs)
    diff1 = c6 - c4
    diff2 = c8 - c6
    assert np.isclose(diff1, diff2, rtol=1e-10)


def test_plate_capacitor_capacitance_analytical_monotonicity() -> None:
    """Test that plate_capacitor_capacitance_analytical behaves monotonically."""
    width = 5.0
    gap = 7.0
    ep_r = 10.0

    # Monotonicity with length (increases)
    c_len1 = plate_capacitor_capacitance_analytical(
        length=10.0, width=width, gap=gap, ep_r=ep_r
    )
    c_len2 = plate_capacitor_capacitance_analytical(
        length=20.0, width=width, gap=gap, ep_r=ep_r
    )
    assert c_len2 > c_len1

    # Monotonicity with gap (decreases)
    c_gap1 = plate_capacitor_capacitance_analytical(
        length=10.0, width=width, gap=5.0, ep_r=ep_r
    )
    c_gap2 = plate_capacitor_capacitance_analytical(
        length=10.0, width=width, gap=10.0, ep_r=ep_r
    )
    assert c_gap1 > c_gap2


def test_plate_capacitor_capacitance_analytical_consistency() -> None:
    """Test that plate_capacitor_capacitance_analytical yields expected physically-reasonable values."""
    # Parameters roughly from standard component designs
    c = plate_capacitor_capacitance_analytical(
        length=26.0, width=5.0, gap=7.0, ep_r=11.7
    )

    c_ff = float(c) * 1e15
    # The expected capacitance for this geometry is ~2 fF.
    assert np.isclose(c_ff, 2.0734, rtol=1e-3)
