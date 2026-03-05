"""Tests for qpdk.models.generic module - LC resonator models."""

from typing import final

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from hypothesis import assume, given, settings

from qpdk.models.generic import lc_resonator, lc_resonator_coupled, nxn

from .base import TwoPortModelTestSuite

MAX_EXAMPLES = 20


@final
class TestLCResonator(TwoPortModelTestSuite):
    """Tests for lc_resonator model."""

    model_function = lc_resonator
    freq_range = (1e9, 30e9)

    def get_model_kwargs(self) -> dict:
        """Get model-specific keyword arguments."""
        return {"capacitance": 100e-15, "inductance": 1e-9}

    def test_resonance_frequency(self) -> None:
        """Test that the resonance frequency matches the expected value f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        f = jnp.linspace(f_r_expected * 0.8, f_r_expected * 1.2, 1000)
        result = lc_resonator(f=f, capacitance=C, inductance=L)

        s21 = result[("o2", "o1")]
        s21_mag = jnp.abs(s21)
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.01, (
            f"Resonance frequency {float(f_observed) / 1e9:.3f} GHz differs from "
            f"expected {f_r_expected / 1e9:.3f} GHz by {relative_error * 100:.2f}%"
        )

    def test_at_resonance_s21_minimum(self) -> None:
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

    def test_at_resonance_s11_maximum(self) -> None:
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

    def test_grounded(self) -> None:
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
    def test_resonance_with_hypothesis(self, L: float, C: float) -> None:
        """Test resonance frequency with random valid L and C values.

        Args:
            L: Inductance in Henries
            C: Capacitance in Farads
        """
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

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


@final
class TestLCResonatorCoupled(TwoPortModelTestSuite):
    """Tests for lc_resonator_coupled model."""

    model_function = lc_resonator_coupled
    freq_range = (1e9, 30e9)

    def get_model_kwargs(self) -> dict:
        """Get model-specific keyword arguments."""
        return {"coupling_capacitance": 10e-15, "coupling_inductance": 1e-9}

    def test_with_capacitance_only(self) -> None:
        """Test coupled resonator with only coupling capacitance.

        Note: When coupling_inductance=0, the inductor acts as a short (Z=0),
        which effectively bypasses the coupling network. This test now checks
        that the model executes correctly and maintains passivity, rather than
        expecting a change from the base resonator.
        """
        f = jnp.linspace(1e9, 30e9, 100)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=0.0
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated (col 1): max total power = {jnp.max(total_power)}"
        )

    def test_with_inductance_only(self) -> None:
        """Test coupled resonator with only coupling inductance."""
        f = jnp.linspace(1e9, 30e9, 100)
        result_no_coupling = lc_resonator(f=f)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=0.0, coupling_inductance=1e-9
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        for key in result_no_coupling:
            assert not jnp.allclose(result_no_coupling[key], result[key], atol=1e-6), (
                f"Coupling inductor had no effect on {key}"
            )

    def test_with_both_coupling_elements(self) -> None:
        """Test coupled resonator with both coupling capacitance and inductance."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = lc_resonator_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=1e-9
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        for key, value in result.items():
            assert jnp.all(jnp.isfinite(value)), (
                f"All values for {key} should be finite"
            )

    def test_resonance_frequency(self) -> None:
        """Test that the main resonance frequency is still approximately f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        f = jnp.linspace(f_r_expected * 0.7, f_r_expected * 1.3, 1000)
        result = lc_resonator_coupled(
            f=f,
            capacitance=C,
            inductance=L,
            coupling_capacitance=1e-15,
        )

        s21_mag = jnp.abs(result[("o2", "o1")])
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.1, (
            f"Main resonance at {float(f_observed) / 1e9:.3f} GHz differs from "
            f"expected {f_r_expected / 1e9:.3f} GHz by {relative_error * 100:.2f}%"
        )

    def test_grounded(self) -> None:
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
    def test_with_hypothesis(self, coupling_C: float, coupling_L: float) -> None:
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

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Should have 4 S-parameters"

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6), "Passivity violated (col 1)"

        s12 = result[("o1", "o2")]
        s22 = result[("o2", "o2")]
        total_power_col2 = jnp.abs(s12) ** 2 + jnp.abs(s22) ** 2
        assert jnp.all(total_power_col2 <= 1.0 + 1e-6), "Passivity violated (col 2)"


@final
class TestNxN:
    """Tests for nxn model."""

    def test_n_ports_assignment(self) -> None:
        """Test that nxn model has the correct number of ports."""
        f = jnp.array([1e9])
        for n in range(1, 6):
            # Sum of ports = n
            result = nxn(f=f, west=n, east=0, north=0, south=0)
            assert isinstance(result, dict)
            # An N-port model has N*N S-parameters
            # Let's check the number of distinct port names in the keys
            ports = set()
            for p1, p2 in result:
                ports.add(p1)
                ports.add(p2)
            assert len(ports) == n, f"Expected {n} ports, got {len(ports)}"

    def test_passivity(self) -> None:
        """Test that nxn model is passive."""
        f = jnp.linspace(1e9, 10e9, 10)
        n = 4
        result = nxn(f=f, west=1, east=1, north=1, south=1)

        for j in range(1, n + 1):
            total_power = jnp.zeros_like(f)
            for i in range(1, n + 1):
                s_ij = result[(f"o{i}", f"o{j}")]
                total_power += jnp.abs(s_ij) ** 2
            assert jnp.all(total_power <= 1.0 + 1e-6), (
                f"Passivity violated for port o{j}: max power = {jnp.max(total_power)}"
            )

    def test_reciprocity(self) -> None:
        """Test that nxn model is reciprocal."""
        f = jnp.array([1e9, 5e9, 10e9])
        n = 3
        result = nxn(f=f, west=1, east=1, north=1, south=0)

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                s_ij = result[(f"o{i}", f"o{j}")]
                s_ji = result[(f"o{j}", f"o{i}")]
                assert jnp.allclose(s_ij, s_ji, atol=1e-10), (
                    f"Reciprocity violated between o{i} and o{j}"
                )

    @given(
        west=st.integers(min_value=0, max_value=5),
        east=st.integers(min_value=0, max_value=5),
        north=st.integers(min_value=0, max_value=5),
        south=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_with_hypothesis(
        self, west: int, east: int, north: int, south: int
    ) -> None:
        """Test nxn model with random port counts using hypothesis."""
        n = west + east + north + south
        assume(n > 0)

        f = jnp.array([1e9, 10e9])
        result = nxn(f=f, west=west, east=east, north=north, south=south)

        # Check port count by looking at unique port names in S-parameter keys
        ports = set()
        for p1, p2 in result:
            ports.add(p1)
            ports.add(p2)
        assert len(ports) == n, f"Expected {n} ports, got {len(ports)} ({ports})"

        # Verify passivity for the first port (o1)
        total_power = jnp.zeros_like(f)
        for i in range(1, n + 1):
            s_i1 = result[(f"o{i}", "o1")]
            total_power += jnp.abs(s_i1) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6), (
            f"Passivity violated for port o1 with N={n}: max power = {jnp.max(total_power)}"
        )
