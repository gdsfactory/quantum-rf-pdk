"""Tests for qpdk.models.capacitor module."""

from typing import cast, final

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_array_less

from qpdk.models.capacitor import (
    interdigital_capacitor,
    interdigital_capacitor_capacitance_analytical,
    plate_capacitor,
    plate_capacitor_capacitance_analytical,
)

from .base import TwoPortModelTestSuite

# Bounds are set to physically reasonable values to avoid numerical issues
permittivities = st.floats(min_value=1.0, max_value=20.0)


@st.composite
def invalid_plate_geometry(draw: st.DrawFn) -> dict[str, float]:
    """Generate plate capacitor geometry with one non-positive or non-finite dimension."""
    dims = {
        "length": draw(st.floats(min_value=1.0, max_value=500.0)),
        "width": draw(st.floats(min_value=0.1, max_value=50.0)),
        "gap": draw(st.floats(min_value=0.1, max_value=50.0)),
    }
    key = draw(st.sampled_from(list(dims)))
    dims[key] = draw(
        st.one_of(
            st.floats(max_value=0.0), st.just(float("inf")), st.just(float("nan"))
        )
    )
    return dims


@st.composite
def invalid_interdigital_geometry(draw: st.DrawFn) -> dict:
    """Generate interdigital capacitor arguments with one invalid dimension."""
    dims = {
        "finger_length": draw(st.floats(min_value=1.0, max_value=100.0)),
        "finger_gap": draw(st.floats(min_value=0.1, max_value=20.0)),
        "thickness": draw(st.floats(min_value=0.1, max_value=20.0)),
    }
    if draw(st.booleans()):
        # Corrupt one geometric dimension
        key = draw(st.sampled_from(list(dims)))
        dims[key] = draw(
            st.one_of(
                st.floats(max_value=0.0), st.just(float("inf")), st.just(float("nan"))
            )
        )
        dims["fingers"] = draw(st.integers(min_value=2, max_value=100))
    else:
        # Invalid finger count: below 2, or fractional (>= 2 but not an integer)
        dims["fingers"] = draw(
            st.one_of(
                st.integers(min_value=-1000, max_value=1),
                st.floats(min_value=2.0, max_value=100.0).filter(
                    lambda x: not float(x).is_integer()
                ),
            )
        )
    return dims


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
        s11_n2 = jnp.abs(result_n2["o1", "o1"])
        s11_n4 = jnp.abs(result_n4["o1", "o1"])

        # At 5GHz, N=4 should have more capacitance than N=2
        assert_array_less(s11_n4, s11_n2 + 1e-10)


def test_interdigital_capacitor_scaling() -> None:
    """Test that interdigital_capacitor capacitance scales correctly."""
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


@settings(deadline=None)
@given(
    geometry=invalid_plate_geometry(),
    ep_r=permittivities,
)
def test_plate_capacitor_invalid_geometry_raises(
    geometry: dict[str, float], ep_r: float
) -> None:
    """Non-positive or non-finite geometry raises ValueError instead of returning inf/NaN."""
    with pytest.raises(ValueError, match="positive and finite"):
        plate_capacitor_capacitance_analytical(ep_r=ep_r, **geometry)


@settings(deadline=None)
@given(
    geometry=invalid_interdigital_geometry(),
    ep_r=permittivities,
)
def test_interdigital_capacitor_invalid_geometry_raises(
    geometry: dict, ep_r: float
) -> None:
    """Invalid fingers or geometry raises ValueError instead of returning NaN."""
    with pytest.raises(ValueError, match=r"positive and finite|fingers"):
        interdigital_capacitor_capacitance_analytical(ep_r=ep_r, **geometry)


@pytest.mark.parametrize("bad_fingers", [1, 0, float("inf"), float("nan")])
def test_interdigital_capacitor_invalid_fingers_raises(bad_fingers: float) -> None:
    """Non-finite or <2 finger counts raise ValueError.

    The non-finite cases cannot be generated by the hypothesis integer strategy,
    so they are pinned here explicitly.
    """
    with pytest.raises(ValueError, match="fingers"):
        interdigital_capacitor_capacitance_analytical(
            fingers=cast(int, bad_fingers),
            finger_length=20.0,
            finger_gap=2.0,
            thickness=5.0,
            ep_r=10.0,
        )


@pytest.mark.parametrize(
    "bad_fingers", [2.5, jnp.array([2, 2.5]), jnp.array([1.5, 3.0])]
)
def test_interdigital_capacitor_fractional_fingers_raises(bad_fingers) -> None:
    """Fractional finger counts raise ValueError.

    The docstring promises a finite integer >= 2; non-integral values
    (scalar or array element) previously computed garbage silently.
    """
    with pytest.raises(ValueError, match="fingers"):
        interdigital_capacitor_capacitance_analytical(
            fingers=bad_fingers,
            finger_length=20.0,
            finger_gap=2.0,
            thickness=5.0,
            ep_r=10.0,
        )


def test_interdigital_capacitor_integral_float_fingers_passes() -> None:
    """Integral float finger counts (e.g. 4.0) are accepted as integers."""
    c = interdigital_capacitor_capacitance_analytical(
        fingers=cast(int, 4.0),
        finger_length=20.0,
        finger_gap=2.0,
        thickness=5.0,
        ep_r=10.0,
    )
    assert bool(jnp.all(jnp.isfinite(c)))


def test_capacitor_analytical_broadcasting() -> None:
    """Broadcast inputs through the public wrappers give finite arrays of expected shape."""
    gap = jnp.geomspace(1.0, 20.0, 5)[:, None]
    length = jnp.linspace(10.0, 500.0, 100)[None, :]

    c_plate = jnp.asarray(
        plate_capacitor_capacitance_analytical(
            length=length, width=10.0, gap=gap, ep_r=11.7
        )
    )
    assert c_plate.shape == (5, 100)
    assert bool(jnp.all(jnp.isfinite(c_plate)))

    fingers = jnp.arange(2, 11, 2)[:, None]
    finger_length = jnp.linspace(10.0, 100.0, 100)[None, :]

    c_idc = jnp.asarray(
        interdigital_capacitor_capacitance_analytical(
            fingers=fingers,
            finger_length=finger_length,
            finger_gap=2.0,
            thickness=5.0,
            ep_r=11.7,
        )
    )
    assert c_idc.shape == (5, 100)
    assert bool(jnp.all(jnp.isfinite(c_idc)))


@final
class TestCapacitorSaxJitComposability:
    """Registered SAX capacitor models must remain jit-composable.

    Regression tests: these models previously called the eagerly-validating
    analytical wrappers, which raised `TracerBoolConversionError` under
    `jax.jit`/`jax.grad`. They now call the jitted cores directly.
    """

    f: jax.Array = jnp.linspace(4e9, 8e9, 11)

    def test_plate_capacitor_jit_grad_traced_length(self) -> None:
        """jit/grad over traced pad length succeeds with finite gradient."""

        def loss(length: float) -> jax.Array:
            s = plate_capacitor(f=self.f, length=length)
            return jnp.abs(s["o1", "o2"][0])

        value = jax.jit(loss)(jnp.array(26.0))
        assert bool(jnp.isfinite(value))
        grad = jax.jit(jax.grad(loss))(jnp.array(26.0))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_interdigital_capacitor_jit_grad_traced_finger_length(self) -> None:
        """jit/grad over traced finger length succeeds with finite gradient."""

        def loss(finger_length: float) -> jax.Array:
            s = interdigital_capacitor(f=self.f, finger_length=finger_length)
            return jnp.abs(s["o1", "o2"][0])

        value = jax.jit(loss)(jnp.array(20.0))
        assert bool(jnp.isfinite(value))
        grad = jax.jit(jax.grad(loss))(jnp.array(20.0))
        assert bool(jnp.all(jnp.isfinite(grad)))
