"""Tests for qpdk.models.waveguides module."""

from typing import final

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import assume, given, settings

from qpdk.models.waveguides import straight
from qpdk.tech import coplanar_waveguide

from .base import TwoPortModelTestSuite

MAX_EXAMPLES = 20


@final
class TestStraightWaveguide(TwoPortModelTestSuite):
    """Unit and integration tests for straight waveguide model."""

    model_function = straight

    def get_model_kwargs(self) -> dict:
        """Get model-specific keyword arguments."""
        return {"length": 1000}

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        length=st.floats(min_value=10, max_value=50000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_passivity_hypothesis(self, f_center: float, length: float) -> None:
        """Test that the waveguide satisfies passivity (energy conservation).

        For a passive two-port network: |S11|^2 + |S21|^2 <= 1

        Args:
            f_center: Center frequency in Hz
            length: Waveguide length in µm
        """
        f = jnp.linspace(f_center * 0.9, f_center * 1.1, 10)
        result = straight(f=f, length=length)

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated: max total power = {jnp.max(total_power)}"
        )

    @given(
        length1=st.floats(min_value=100, max_value=10000),
        length2=st.floats(min_value=100, max_value=10000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_length_effect(self, length1: float, length2: float) -> None:
        """Test that longer waveguides have more attenuation.

        Args:
            length1: First waveguide length in µm
            length2: Second waveguide length in µm
        """
        assume(abs(length2 - length1) > 100)

        f = jnp.array([5e9])
        result1 = straight(f=f, length=length1)
        result2 = straight(f=f, length=length2)

        transmission1 = jnp.abs(result1[("o2", "o1")])[0]
        transmission2 = jnp.abs(result2[("o2", "o1")])[0]

        if length2 > length1:
            assert transmission2 <= transmission1 + 1e-10, (
                f"Longer waveguide should have lower transmission: "
                f"L1={length1}, |S21|1={transmission1}, "
                f"L2={length2}, |S21|2={transmission2}"
            )

    def test_frequency_sweep(self) -> None:
        """Test straight waveguide across typical superconducting RF frequency range."""
        f = jnp.linspace(0.5e9, 10e9, 100)
        length = 5000  # 5 mm

        result = straight(f=f, length=length)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result[("o2", "o1")]) == 100, "Should have 100 frequency points"

        s21 = result[("o2", "o1")]
        assert jnp.all(jnp.isfinite(s21)), "All S21 values should be finite"
        assert jnp.all(jnp.abs(s21) <= 1.0 + 1e-10), (
            "All |S21| values should be <= 1 (with numerical tolerance)"
        )

    def test_custom_cross_section(self) -> None:
        """Test straight waveguide with custom media parameters."""
        custom_cross_section = coplanar_waveguide(width=20, gap=10)

        f = jnp.array([5e9, 6e9])
        result = straight(f=f, length=1000, cross_section=custom_cross_section)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result[("o2", "o1")]) == 2, "Should have 2 frequency points"

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_zero_length(self) -> None:
        """Test straight waveguide with zero length (through connection)."""
        f = jnp.array([5e9])
        result = straight(f=f, length=0)

        s21 = result[("o2", "o1")]
        transmission = jnp.abs(s21)[0]

        assert transmission > 0.99, (
            f"Zero length should have ~perfect transmission, got {transmission}"
        )
