"""Tests for qpdk.models.capacitor module."""

from typing import TYPE_CHECKING, final

import jax.numpy as jnp
import numpy as np

from qpdk.models.capacitor import interdigital_capacitor, plate_capacitor
from tests.models.base import TwoPortModelTestSuite

if TYPE_CHECKING:
    pass


@final
class TestPlateCapacitor(TwoPortModelTestSuite):
    """Test plate_capacitor model."""

    model_function = staticmethod(plate_capacitor)

    def get_model_kwargs(self) -> dict:
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
        assert jnp.all(s11_n4 <= s11_n2 + 1e-10)


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
