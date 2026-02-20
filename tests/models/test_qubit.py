"""Tests for qpdk.models.qubit module."""

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

from qpdk.models.qubit import lc_resonator_capacitive, lc_resonator_inductive

MAX_EXAMPLES = 20


class TestLCResonatorCapacitive:
    """Unit and integration tests for capacitively coupled LC resonator model."""

    def test_lc_resonator_capacitive_default_parameters(self) -> None:
        """Test LC resonator with default parameters."""
        result = lc_resonator_capacitive()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"
        assert ("o2", "o1") in result, "Should have S21 parameter"
        assert ("o2", "o2") in result, "Should have S22 parameter"

    def test_lc_resonator_capacitive_returns_stype(self) -> None:
        """Test that lc_resonator_capacitive returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

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

    def test_lc_resonator_capacitive_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_lc_resonator_capacitive_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity: S12 should equal S21
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_lc_resonator_capacitive_passivity(self) -> None:
        """Test that the LC resonator is passive (|S11|^2 + |S21|^2 <= 1)."""
        f = jnp.linspace(1e9, 10e9, 100)
        result = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        # Check passivity condition
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        max_power = jnp.max(power)

        # Allow small numerical tolerance
        assert max_power <= 1.0 + 1e-6, (
            f"Maximum power {max_power} exceeds 1 (passivity violation)"
        )

    def test_lc_resonator_capacitive_resonance_peak(self) -> None:
        """Test that the LC resonator exhibits a resonance peak."""
        # Set up parameters for observable resonance
        inductance = 1e-9  # 1 nH
        capacitance = 100e-15  # 100 fF
        # Expected resonance: f_r = 1/(2*pi*sqrt(LC)) ~ 1.59 GHz

        f = jnp.linspace(0.5e9, 3e9, 200)
        result = lc_resonator_capacitive(
            f=f,
            inductance=inductance,
            capacitance=capacitance,
            coupling_capacitance=10e-15,
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        # At resonance, transmission (S21) should have a peak/minimum in reflection (S11)
        # Find the frequency with maximum transmission
        max_s21_idx = jnp.argmax(jnp.abs(s21))
        min_s11_idx = jnp.argmin(jnp.abs(s11))

        # The indices should be close (within 10% of frequency range)
        assert abs(max_s21_idx - min_s11_idx) < len(f) * 0.1, (
            "S21 peak and S11 minimum should occur at similar frequencies"
        )

    def test_lc_resonator_capacitive_single_frequency(self) -> None:
        """Test LC resonator with a single frequency point."""
        f = jnp.array([5e9])
        result = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        for key, value in result.items():
            assert len(value) == 1, f"Expected single value for {key}, got {len(value)}"

    @given(
        inductance=st.floats(min_value=0.1e-9, max_value=10e-9),
        capacitance=st.floats(min_value=10e-15, max_value=200e-15),
        coupling_capacitance=st.floats(min_value=1e-15, max_value=50e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_lc_resonator_capacitive_hypothesis(
        self,
        inductance: float,
        capacitance: float,
        coupling_capacitance: float,
    ) -> None:
        """Property-based test with random valid parameters."""
        f = jnp.array([5e9])
        result = lc_resonator_capacitive(
            f=f,
            inductance=inductance,
            capacitance=capacitance,
            coupling_capacitance=coupling_capacitance,
        )

        # Check basic properties
        assert isinstance(result, dict)
        assert len(result) == 4  # Four S-parameters for 2-port

        # Check reciprocity
        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]
        assert jnp.allclose(s12, s21, atol=1e-10)

        # Check passivity
        s11 = result[("o1", "o1")]
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(power <= 1.0 + 1e-6)


class TestLCResonatorInductive:
    """Unit and integration tests for inductively coupled LC resonator model."""

    def test_lc_resonator_inductive_default_parameters(self) -> None:
        """Test LC resonator with default parameters."""
        result = lc_resonator_inductive()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"
        assert ("o2", "o1") in result, "Should have S21 parameter"
        assert ("o2", "o2") in result, "Should have S22 parameter"

    def test_lc_resonator_inductive_returns_stype(self) -> None:
        """Test that lc_resonator_inductive returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

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

    def test_lc_resonator_inductive_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

        # Check all S-parameter arrays have correct length
        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_lc_resonator_inductive_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        # Check reciprocity: S12 should equal S21
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_lc_resonator_inductive_passivity(self) -> None:
        """Test that the LC resonator is passive (|S11|^2 + |S21|^2 <= 1)."""
        f = jnp.linspace(1e9, 10e9, 100)
        result = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        # Check passivity condition
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        max_power = jnp.max(power)

        # Allow small numerical tolerance
        assert max_power <= 1.0 + 1e-6, (
            f"Maximum power {max_power} exceeds 1 (passivity violation)"
        )

    def test_lc_resonator_inductive_single_frequency(self) -> None:
        """Test LC resonator with a single frequency point."""
        f = jnp.array([5e9])
        result = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        for key, value in result.items():
            assert len(value) == 1, f"Expected single value for {key}, got {len(value)}"

    @given(
        inductance=st.floats(min_value=0.5e-9, max_value=10e-9),
        capacitance=st.floats(min_value=10e-15, max_value=200e-15),
        coupling_inductance=st.floats(min_value=0.5e-9, max_value=5e-9),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_lc_resonator_inductive_hypothesis(
        self,
        inductance: float,
        capacitance: float,
        coupling_inductance: float,
    ) -> None:
        """Property-based test with random valid parameters."""
        # Mutual inductance should be less than the smaller of the two inductances
        mutual_inductance = min(inductance, coupling_inductance) * 0.1

        f = jnp.array([5e9])
        result = lc_resonator_inductive(
            f=f,
            inductance=inductance,
            capacitance=capacitance,
            mutual_inductance=mutual_inductance,
            coupling_inductance=coupling_inductance,
        )

        # Check basic properties
        assert isinstance(result, dict)
        assert len(result) == 4  # Four S-parameters for 2-port

        # Check reciprocity
        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]
        assert jnp.allclose(s12, s21, atol=1e-10)

        # Check passivity
        s11 = result[("o1", "o1")]
        power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(power <= 1.0 + 1e-6)


class TestLCResonatorComparison:
    """Compare capacitive and inductive coupling models."""

    def test_both_models_are_passive(self) -> None:
        """Test that both coupling types are passive."""
        f = jnp.linspace(1e9, 10e9, 100)

        # Capacitive coupling
        s_cap = lc_resonator_capacitive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            coupling_capacitance=10e-15,
        )

        # Inductive coupling
        s_ind = lc_resonator_inductive(
            f=f,
            inductance=1e-9,
            capacitance=50e-15,
            mutual_inductance=0.1e-9,
            coupling_inductance=1e-9,
        )

        # Check passivity for both
        for s, name in [(s_cap, "capacitive"), (s_ind, "inductive")]:
            s11 = s[("o1", "o1")]
            s21 = s[("o2", "o1")]
            power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
            max_power = jnp.max(power)
            assert max_power <= 1.0 + 1e-6, (
                f"{name} coupling: Maximum power {max_power} exceeds 1"
            )
