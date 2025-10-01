"""Tests for qpdk.models.waveguides.straight function."""

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import assume, given, settings

from qpdk.models.media import cpw_media_skrf
from qpdk.models.waveguides import straight

MAX_EXAMPLES = 20


class TestStraightWaveguide:
    """Unit and integration tests for straight waveguide model."""

    def test_straight_default_parameters(self) -> None:
        """Test straight waveguide with default parameters."""
        result = straight()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"
        assert ("o2", "o1") in result, "Should have S21 parameter"
        assert ("o2", "o2") in result, "Should have S22 parameter"

    def test_straight_returns_stype(self) -> None:
        """Test that straight returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = straight(f=f, length=1000)

        # Check it's a dictionary with the expected structure
        assert isinstance(result, dict), "Result should be a dictionary"

        # Check all required S-parameter keys are present
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

        # Check all values are arrays
        for key, value in result.items():
            assert hasattr(value, "__len__"), f"Value for {key} should be array-like"

    def test_straight_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = straight(f=f, length=2000)

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_straight_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = straight(f=f, length=1500)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity: S12 should equal S21
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_straight_single_frequency(self) -> None:
        """Test straight waveguide with a single frequency point."""
        f = jnp.array([5e9])
        result = straight(f=f, length=1000)

        assert isinstance(result, dict), "Result should be a dictionary"
        for key, value in result.items():
            assert len(value) == 1, f"Expected single value for {key}, got {len(value)}"

    @given(
        n_freq=st.integers(min_value=1, max_value=100),
        f_min=st.floats(min_value=0.5e9, max_value=5e9),
        f_max=st.floats(min_value=5e9, max_value=10e9),
        length=st.floats(min_value=1, max_value=100000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_straight_with_hypothesis(
        self, n_freq: int, f_min: float, f_max: float, length: float
    ) -> None:
        """Test straight waveguide with random valid parameters using hypothesis.

        Args:
            n_freq: Number of frequency points
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
            length: Waveguide length in µm
        """
        assume(f_max > f_min)

        f = jnp.linspace(f_min, f_max, n_freq)
        result = straight(f=f, length=length)

        # Verify structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        # Verify shapes
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"S-parameter {key} length should match frequency array length"
            )

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        length=st.floats(min_value=10, max_value=50000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_straight_passivity(self, f_center: float, length: float) -> None:
        """Test that the waveguide satisfies passivity (energy conservation).

        For a passive two-port network: |S11|^2 + |S21|^2 <= 1

        Args:
            f_center: Center frequency in Hz
            length: Waveguide length in µm
        """
        # Use a small frequency range around the center
        f = jnp.linspace(f_center * 0.9, f_center * 1.1, 10)
        result = straight(f=f, length=length)

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        # Check passivity constraint
        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        # Allow small numerical tolerance
        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated: max total power = {jnp.max(total_power)}"
        )

    @given(
        length1=st.floats(min_value=100, max_value=10000),
        length2=st.floats(min_value=100, max_value=10000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_straight_length_effect(self, length1: float, length2: float) -> None:
        """Test that longer waveguides have more attenuation.

        Args:
            length1: First waveguide length in µm
            length2: Second waveguide length in µm
        """
        assume(abs(length2 - length1) > 100)  # Ensure meaningful difference

        # Use a typical superconducting frequency
        f = jnp.array([5e9])

        result1 = straight(f=f, length=length1)
        result2 = straight(f=f, length=length2)

        transmission1 = jnp.abs(result1[("o2", "o1")])[0]
        transmission2 = jnp.abs(result2[("o2", "o1")])[0]

        # The longer waveguide should have equal or lower transmission
        # (accounting for numerical precision)
        if length2 > length1:
            assert transmission2 <= transmission1 + 1e-10, (
                f"Longer waveguide should have lower transmission: "
                f"L1={length1}, |S21|1={transmission1}, "
                f"L2={length2}, |S21|2={transmission2}"
            )

    def test_straight_frequency_sweep(self) -> None:
        """Test straight waveguide across typical superconducting RF frequency range."""
        # Typical superconducting frequency range: 0.5 to 10 GHz
        f = jnp.linspace(0.5e9, 10e9, 100)
        length = 5000  # 5 mm

        result = straight(f=f, length=length)

        # Basic sanity checks
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result[("o2", "o1")]) == 100, "Should have 100 frequency points"

        # Check that transmission is reasonable (not NaN or infinite)
        s21 = result[("o2", "o1")]
        assert jnp.all(jnp.isfinite(s21)), "All S21 values should be finite"
        # Allow small numerical tolerance for floating point precision
        assert jnp.all(jnp.abs(s21) <= 1.0 + 1e-10), (
            "All |S21| values should be <= 1 (with numerical tolerance)"
        )

    def test_straight_custom_media(self) -> None:
        """Test straight waveguide with custom media parameters."""
        # Create custom CPW media with different dimensions
        custom_media = cpw_media_skrf(width=20, gap=10)

        f = jnp.array([5e9, 6e9])
        result = straight(f=f, length=1000, media=custom_media)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result[("o2", "o1")]) == 2, "Should have 2 frequency points"

        # Should still satisfy reciprocity
        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_straight_zero_length(self) -> None:
        """Test straight waveguide with zero length (through connection)."""
        f = jnp.array([5e9])
        result = straight(f=f, length=0)

        # Zero length should be nearly perfect transmission
        s21 = result[("o2", "o1")]
        transmission = jnp.abs(s21)[0]

        # Should be very close to 1 (perfect transmission)
        assert transmission > 0.99, (
            f"Zero length should have ~perfect transmission, got {transmission}"
        )
