"""Tests for qpdk.models.generic module - LC resonator models."""

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from hypothesis import assume, given, settings

from qpdk.models.generic import lc_resonator, lc_resonator_coupled

MAX_EXAMPLES = 20


class TestLCResonator:
    """Tests for lc_resonator model."""

    def test_lc_resonator_default_parameters(self) -> None:
        """Test lc_resonator with default parameters."""
        result = lc_resonator()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"
        assert ("o2", "o1") in result, "Should have S21 parameter"
        assert ("o2", "o2") in result, "Should have S22 parameter"

    def test_lc_resonator_returns_stype(self) -> None:
        """Test that lc_resonator returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = lc_resonator(f=f, capacitance=100e-15, inductance=1e-9)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

        for key, value in result.items():
            assert hasattr(value, "__len__"), f"Value for {key} should be array-like"

    def test_lc_resonator_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = lc_resonator(f=f)

        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_lc_resonator_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 20e9, 50)
        result = lc_resonator(f=f)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_lc_resonator_passivity(self) -> None:
        """Test that the resonator satisfies passivity (energy conservation)."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = lc_resonator(f=f)

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated (col 1): max total power = {jnp.max(total_power)}"
        )

        s12 = result[("o1", "o2")]
        s22 = result[("o2", "o2")]
        total_power_col2 = jnp.abs(s12) ** 2 + jnp.abs(s22) ** 2
        assert jnp.all(total_power_col2 <= 1.0 + 1e-6), (
            f"Passivity violated (col 2): max total power = {jnp.max(total_power_col2)}"
        )

    def test_lc_resonator_resonance_frequency(self) -> None:
        """Test that the resonance frequency matches the expected value f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        # Fine frequency sweep around resonance
        f = jnp.linspace(f_r_expected * 0.8, f_r_expected * 1.2, 1000)
        result = lc_resonator(f=f, capacitance=C, inductance=L)

        # At resonance, |S21| should be minimum (parallel LC has infinite impedance)
        s21 = result[("o2", "o1")]
        s21_mag = jnp.abs(s21)
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        # Check resonance frequency is within 1% of expected
        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.01, (
            f"Resonance frequency {float(f_observed) / 1e9:.3f} GHz differs from "
            f"expected {f_r_expected / 1e9:.3f} GHz by {relative_error * 100:.2f}%"
        )

    def test_lc_resonator_at_resonance_s21_minimum(self) -> None:
        """Test that |S21| approaches zero at resonance for parallel LC."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        f = jnp.array([f_r_expected])
        result = lc_resonator(f=f, capacitance=C, inductance=L)

        s21_at_resonance = float(jnp.abs(result[("o2", "o1")])[0])
        assert s21_at_resonance < 0.01, (
            f"|S21| at resonance should be near zero, got {s21_at_resonance}"
        )

    def test_lc_resonator_at_resonance_s11_maximum(self) -> None:
        """Test that |S11| approaches 1 at resonance for parallel LC."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        f = jnp.array([f_r_expected])
        result = lc_resonator(f=f, capacitance=C, inductance=L)

        s11_at_resonance = float(jnp.abs(result[("o1", "o1")])[0])
        assert s11_at_resonance > 0.99, (
            f"|S11| at resonance should be near 1, got {s11_at_resonance}"
        )

    def test_lc_resonator_grounded(self) -> None:
        """Test that grounded LC resonator has expected structure."""
        f = jnp.array([5e9, 10e9, 15e9])
        result = lc_resonator(f=f, grounded=True)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

        # Second port is grounded, so S22 should be -1 (short reflection)
        s22 = result[("o2", "o2")]
        assert jnp.allclose(s22, -1.0, atol=1e-10), (
            f"S22 should be -1 for grounded port, got {s22}"
        )

        # S21 should be zero (no transmission through ground)
        s21 = result[("o2", "o1")]
        assert jnp.allclose(s21, 0.0, atol=1e-10), (
            f"S21 should be 0 for grounded port, got {s21}"
        )

    @given(
        L=st.floats(min_value=0.1e-9, max_value=10e-9),
        C=st.floats(min_value=10e-15, max_value=1000e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_lc_resonator_resonance_with_hypothesis(self, L: float, C: float) -> None:
        """Test resonance frequency with random valid L and C values.

        Args:
            L: Inductance in Henries
            C: Capacitance in Farads
        """
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        # Skip if resonance frequency is outside reasonable range
        assume(1e9 <= f_r_expected <= 50e9)

        f = jnp.linspace(f_r_expected * 0.8, f_r_expected * 1.2, 500)
        result = lc_resonator(f=f, capacitance=C, inductance=L)

        s21_mag = jnp.abs(result[("o2", "o1")])
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.02, (
            f"Resonance at {float(f_observed) / 1e9:.3f} GHz differs from "
            f"expected {f_r_expected / 1e9:.3f} GHz by {relative_error * 100:.2f}%"
        )


class TestLCResonatorCoupled:
    """Tests for lc_resonator_coupled model."""

    def test_lc_resonator_coupled_default_parameters(self) -> None:
        """Test lc_resonator_coupled with default parameters (no coupling)."""
        result = lc_resonator_coupled()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"
        assert ("o2", "o1") in result, "Should have S21 parameter"
        assert ("o2", "o2") in result, "Should have S22 parameter"

    def test_lc_resonator_coupled_no_coupling_equals_basic(self) -> None:
        """Test that lc_resonator_coupled with zero coupling equals lc_resonator."""
        f = jnp.array([5e9, 10e9, 15e9])
        C = 100e-15
        L = 1e-9

        result_basic = lc_resonator(f=f, capacitance=C, inductance=L)
        result_coupled = lc_resonator_coupled(
            f=f,
            capacitance=C,
            inductance=L,
            coupling_capacitance=0.0,
            coupling_inductance=0.0,
        )

        for key in result_basic:
            diff = jnp.max(jnp.abs(result_basic[key] - result_coupled[key]))
            assert diff < 1e-10, (
                f"With zero coupling, results should match for {key}, diff={diff}"
            )

    def test_lc_resonator_coupled_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = lc_resonator_coupled(f=f, coupling_capacitance=10e-15)

        for key, value in result.items():
            assert len(value) == n_freq, (
                f"Expected length {n_freq} for {key}, got {len(value)}"
            )

    def test_lc_resonator_coupled_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 20e9, 50)
        result = lc_resonator_coupled(f=f, coupling_capacitance=10e-15)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10, f"S12 and S21 should be equal, max diff: {max_diff}"

    def test_lc_resonator_coupled_passivity(self) -> None:
        """Test that the coupled resonator satisfies passivity."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=1e-9
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated (col 1): max total power = {jnp.max(total_power)}"
        )

        s12 = result[("o1", "o2")]
        s22 = result[("o2", "o2")]
        total_power_col2 = jnp.abs(s12) ** 2 + jnp.abs(s22) ** 2
        assert jnp.all(total_power_col2 <= 1.0 + 1e-6), (
            f"Passivity violated (col 2): max total power = {jnp.max(total_power_col2)}"
        )

    def test_lc_resonator_coupled_with_capacitance_only(self) -> None:
        """Test coupled resonator with only coupling capacitance."""
        f = jnp.linspace(1e9, 30e9, 100)
        result_no_coupling = lc_resonator(f=f)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=0.0
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        # The coupling capacitor should change the S-parameters away from the basic resonator
        for key in result_no_coupling:
            assert not jnp.allclose(result_no_coupling[key], result[key], atol=1e-6), (
                f"Coupling capacitor had no effect on {key}"
            )

    def test_lc_resonator_coupled_with_inductance_only(self) -> None:
        """Test coupled resonator with only coupling inductance."""
        f = jnp.linspace(1e9, 30e9, 100)
        result_no_coupling = lc_resonator(f=f)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=0.0, coupling_inductance=1e-9
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        # The coupling inductor should change the S-parameters away from the basic resonator
        for key in result_no_coupling:
            assert not jnp.allclose(result_no_coupling[key], result[key], atol=1e-6), (
                f"Coupling inductor had no effect on {key}"
            )

    def test_lc_resonator_coupled_with_both_coupling_elements(self) -> None:
        """Test coupled resonator with both coupling capacitance and inductance."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=1e-9
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        # Verify all values are finite
        for key, value in result.items():
            assert jnp.all(jnp.isfinite(value)), (
                f"All values for {key} should be finite"
            )

    def test_lc_resonator_coupled_resonance_frequency(self) -> None:
        """Test that the main resonance frequency is still approximately f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        # With weak coupling, main resonance should be close to expected
        f = jnp.linspace(f_r_expected * 0.7, f_r_expected * 1.3, 1000)
        result = lc_resonator_coupled(
            f=f,
            capacitance=C,
            inductance=L,
            coupling_capacitance=1e-15,  # weak coupling
        )

        s21_mag = jnp.abs(result[("o2", "o1")])
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        # With coupling, resonance may shift, allow 10% tolerance
        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.1, (
            f"Main resonance at {float(f_observed) / 1e9:.3f} GHz differs from "
            f"expected {f_r_expected / 1e9:.3f} GHz by {relative_error * 100:.2f}%"
        )

    def test_lc_resonator_coupled_grounded(self) -> None:
        """Test coupled resonator with grounded base resonator."""
        f = jnp.array([5e9, 10e9, 15e9])
        result = lc_resonator_coupled(f=f, grounded=True, coupling_capacitance=10e-15)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    @given(
        coupling_C=st.floats(min_value=1e-15, max_value=100e-15),
        coupling_L=st.floats(min_value=0.1e-9, max_value=10e-9),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_lc_resonator_coupled_with_hypothesis(
        self, coupling_C: float, coupling_L: float
    ) -> None:
        """Test coupled resonator with random coupling values.

        Args:
            coupling_C: Coupling capacitance in Farads
            coupling_L: Coupling inductance in Henries
        """
        f = jnp.linspace(1e9, 30e9, 50)
        result = lc_resonator_coupled(
            f=f,
            coupling_capacitance=coupling_C,
            coupling_inductance=coupling_L,
        )

        # Basic structure checks
        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        # Verify passivity
        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6), "Passivity violated (col 1)"

        s12 = result[("o1", "o2")]
        s22 = result[("o2", "o2")]
        total_power_col2 = jnp.abs(s12) ** 2 + jnp.abs(s22) ** 2
        assert jnp.all(total_power_col2 <= 1.0 + 1e-6), "Passivity violated (col 2)"
