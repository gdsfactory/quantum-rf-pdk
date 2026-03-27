"""Tests for qpdk.models.qubit module - Qubit LC resonator models."""

from typing import final

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose, assert_array_less

import qpdk
from qpdk.models.constants import Φ_0, e, h
from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    double_island_transmon,
    double_island_transmon_with_bbox,
    double_island_transmon_with_resonator,
    ec_to_capacitance,
    ej_to_inductance,
    flipmon,
    flipmon_with_bbox,
    flipmon_with_resonator,
    qubit_with_resonator,
    shunted_transmon,
    transmon_coupled,
    transmon_with_resonator,
    xmon_transmon,
)

from .base import TwoPortModelTestSuite

# Ensure PDK is activated for tests that require it (e.g., qubit_with_resonator)
qpdk.PDK.activate()

MAX_EXAMPLES = 20


class TestEcToCapacitance:
    """Tests for ec_to_capacitance helper function."""

    @staticmethod
    def test_typical_value() -> None:
        """Test with typical transmon charging energy."""
        # E_C = 0.2 GHz is typical for transmons
        ec_ghz = 0.2
        C = ec_to_capacitance(ec_ghz)

        # C_Σ = e² / (2 * E_C)
        # For E_C = 0.2 GHz, C_Σ ≈ 96 fF
        expected_C = e**2 / (2 * ec_ghz * 1e9 * h)
        assert np.isclose(C, expected_C, rtol=1e-10)
        # Check reasonable range
        assert 50e-15 < C < 200e-15, f"Capacitance {C * 1e15:.1f} fF out of range"

    @staticmethod
    def test_inverse_relationship() -> None:
        """Test that capacitance decreases as E_C increases."""
        C_low = ec_to_capacitance(0.1)  # Low E_C
        C_high = ec_to_capacitance(0.5)  # High E_C
        assert C_low > C_high

    @given(ec_ghz=st.floats(min_value=0.05, max_value=1.0))
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_capacitance(self, ec_ghz: float) -> None:
        """Test that capacitance is always positive."""
        C = ec_to_capacitance(ec_ghz)
        assert C > 0


class TestEjToInductance:
    """Tests for ej_to_inductance helper function."""

    @staticmethod
    def test_typical_value() -> None:
        """Test with typical transmon Josephson energy."""
        # E_J = 20 GHz is typical for transmons
        ej_ghz = 20.0
        L = ej_to_inductance(ej_ghz)

        # L_J = Φ_0² / (4π² E_J)
        expected_L = Φ_0**2 / (4 * np.pi**2 * ej_ghz * 1e9 * h)
        assert np.isclose(L, expected_L, rtol=1e-10)
        # Check reasonable range (~ 1 nH for 20 GHz)
        assert 0.1e-9 < L < 10e-9, f"Inductance {L * 1e9:.2f} nH out of range"

    @staticmethod
    def test_inverse_relationship() -> None:
        """Test that inductance decreases as E_J increases."""
        L_low = ej_to_inductance(10.0)  # Low E_J
        L_high = ej_to_inductance(40.0)  # High E_J
        assert L_low > L_high

    @given(ej_ghz=st.floats(min_value=5.0, max_value=100.0))
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_inductance(self, ej_ghz: float) -> None:
        """Test that inductance is always positive."""
        L = ej_to_inductance(ej_ghz)
        assert L > 0


class TestCouplingStrengthToCapacitance:
    """Tests for coupling_strength_to_capacitance helper function."""

    @staticmethod
    def test_typical_value() -> None:
        """Test with typical qubit-resonator coupling parameters."""
        # Typical values:
        # g = 0.1 GHz, C_Σ = 80 fF, C_r = 50 fF, ω_q = 5 GHz, ω_r = 7 GHz
        C_c = coupling_strength_to_capacitance(
            g_ghz=0.1,
            c_sigma=80e-15,
            c_r=50e-15,
            f_q_ghz=5.0,
            f_r_ghz=7.0,
        )

        # Should give a few fF coupling capacitance
        assert 0.1e-15 < C_c < 50e-15, (
            f"Coupling capacitance {C_c * 1e15:.2f} fF out of range"
        )

    @staticmethod
    def test_coupling_increases_with_g() -> None:
        """Test that coupling capacitance increases with g."""
        C_c_weak = coupling_strength_to_capacitance(
            g_ghz=0.05,
            c_sigma=80e-15,
            c_r=50e-15,
            f_q_ghz=5.0,
            f_r_ghz=7.0,
        )
        C_c_strong = coupling_strength_to_capacitance(
            g_ghz=0.2,
            c_sigma=80e-15,
            c_r=50e-15,
            f_q_ghz=5.0,
            f_r_ghz=7.0,
        )
        assert C_c_strong > C_c_weak

    @staticmethod
    @given(
        g_ghz=st.floats(min_value=0.01, max_value=0.5),
        c_sigma=st.floats(min_value=10e-15, max_value=200e-15),
        c_r=st.floats(min_value=10e-15, max_value=200e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_coupling_capacitance(
        g_ghz: float, c_sigma: float, c_r: float
    ) -> None:
        """Test that coupling capacitance is always positive."""
        C_c = coupling_strength_to_capacitance(
            g_ghz=g_ghz,
            c_sigma=c_sigma,
            c_r=c_r,
            f_q_ghz=5.0,
            f_r_ghz=7.0,
        )
        assert C_c > 0


@final
class TestDoubleIslandTransmon(TwoPortModelTestSuite):
    """Tests for double_island_transmon model."""

    model_function = staticmethod(double_island_transmon)

    def test_is_ungrounded(self) -> None:
        """Test that double-island transmon has ungrounded behavior."""
        f = self.get_frequency_array(3)
        result = self._call_model(f=f)

        # For ungrounded, S22 should NOT be -1 (which would indicate a short)
        s22 = result["o2", "o2"]
        assert not jnp.allclose(s22, -1.0, atol=1e-6), (
            "Double-island transmon should NOT be grounded"
        )

        # For an ungrounded parallel LC, S22 should equal S11 due to symmetry
        s11 = result["o1", "o1"]
        assert_allclose(
            s11,
            s22,
            atol=1e-10,
            err_msg="S11 and S22 should be equal for symmetric ungrounded LC resonator",
        )

    def test_resonance_frequency(self) -> None:
        """Test that the resonance frequency matches expected f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        # Fine frequency sweep around resonance
        f = jnp.linspace(f_r_expected * 0.8, f_r_expected * 1.2, 1000)
        result = self._call_model(f=f, capacitance=C, inductance=L)

        # At resonance, |S21| should be minimum (parallel LC has infinite impedance)
        s21 = result["o2", "o1"]
        s21_mag = jnp.abs(s21)
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.01


@final
class TestDoubleIslandTransmonWithBbox(TwoPortModelTestSuite):
    """Tests for double_island_transmon_with_bbox wrapper."""

    model_function = staticmethod(double_island_transmon_with_bbox)


@final
class TestFlipmon(TwoPortModelTestSuite):
    """Tests for flipmon wrapper."""

    model_function = staticmethod(flipmon)


@final
class TestFlipmonWithBbox(TwoPortModelTestSuite):
    """Tests for flipmon_with_bbox wrapper."""

    model_function = staticmethod(flipmon_with_bbox)


@final
class TestXmonTransmon(TwoPortModelTestSuite):
    """Tests for xmon_transmon wrapper."""

    model_function = staticmethod(xmon_transmon)


@final
class TestShuntedTransmon(TwoPortModelTestSuite):
    """Tests for shunted_transmon model."""

    model_function = staticmethod(shunted_transmon)

    def test_is_grounded(self) -> None:
        """Test that shunted transmon has grounded behavior."""
        f = self.get_frequency_array(3)
        result = self._call_model(f=f)

        # For grounded, S22 should be -1 (short reflection)
        s22 = result["o2", "o2"]
        assert_allclose(
            s22,
            -1.0,
            atol=1e-10,
            err_msg=f"S22 should be -1 for grounded port, got {s22}",
        )

        # S21 should be zero (no transmission through ground)
        s21 = result["o2", "o1"]
        assert_allclose(s21, 0.0, atol=1e-10)

    @given(
        L=st.floats(min_value=0.1e-9, max_value=10e-9),
        C=st.floats(min_value=10e-15, max_value=1000e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_with_hypothesis(self, L: float, C: float) -> None:
        """Test shunted transmon with random valid L and C values."""
        f = self.get_frequency_array(50)
        result = self._call_model(f=f, capacitance=C, inductance=L)

        assert isinstance(result, dict)
        assert len(result) == 4


@final
class TestTransmonCoupled(TwoPortModelTestSuite):
    """Tests for transmon_coupled model."""

    model_function = staticmethod(transmon_coupled)

    def test_capacitive_coupling(self) -> None:
        """Test coupled transmon with capacitive coupling only."""
        f = self.get_frequency_array(100)
        result = self._call_model(
            f=f, coupling_capacitance=10e-15, coupling_inductance=0.0
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify passivity
        s11 = result["o1", "o1"]
        s21 = result["o2", "o1"]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert_array_less(total_power, 1.0 + 1e-6)

    def test_inductive_coupling(self) -> None:
        """Test coupled transmon with inductive coupling only."""
        f = self.get_frequency_array(100)
        result = self._call_model(
            f=f, coupling_capacitance=0.0, coupling_inductance=1e-9
        )

        assert isinstance(result, dict)
        assert len(result) == 4

    def test_grounded_coupling(self) -> None:
        """Test coupled transmon with grounded qubit."""
        f = self.get_frequency_array(3)
        result = self._call_model(f=f, grounded=True, coupling_capacitance=10e-15)

        assert isinstance(result, dict)
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys

    @given(
        coupling_C=st.floats(min_value=1e-15, max_value=100e-15),
        coupling_L=st.floats(min_value=0.1e-9, max_value=10e-9),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_with_hypothesis(self, coupling_C: float, coupling_L: float) -> None:
        """Test coupled transmon with random coupling values."""
        f = self.get_frequency_array(50)
        result = self._call_model(
            f=f,
            coupling_capacitance=coupling_C,
            coupling_inductance=coupling_L,
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify passivity
        s11 = result["o1", "o1"]
        s21 = result["o2", "o1"]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert_array_less(total_power, 1.0 + 1e-6)


class TestIntegration:
    """Integration tests combining helper functions with qubit models."""

    @staticmethod
    def test_ec_ej_to_qubit_model() -> None:
        """Test that E_C and E_J can be converted to qubit model parameters."""
        # Typical transmon parameters
        ec_ghz = 0.2  # Charging energy
        ej_ghz = 20.0  # Josephson energy

        # Convert to circuit parameters
        C = ec_to_capacitance(ec_ghz)
        L = ej_to_inductance(ej_ghz)

        # Verify reasonable values
        assert 50e-15 < C < 200e-15, f"Capacitance {C * 1e15:.1f} fF out of range"
        assert 0.1e-9 < L < 20e-9, f"Inductance {L * 1e9:.2f} nH out of range"

        # Create qubit models with converted parameters
        f = jnp.linspace(1e9, 20e9, 100)

        result_double = double_island_transmon(f=f, capacitance=C, inductance=L)
        assert isinstance(result_double, dict)

        result_shunted = shunted_transmon(f=f, capacitance=C, inductance=L)
        assert isinstance(result_shunted, dict)

    @staticmethod
    def test_coupled_qubit_with_converted_parameters() -> None:
        """Test coupled qubit with parameters converted from Hamiltonian."""
        # Typical transmon parameters
        ec_ghz = 0.2
        ej_ghz = 20.0
        g_ghz = 0.1  # Coupling strength

        # Convert to circuit parameters
        C = ec_to_capacitance(ec_ghz)
        L = ej_to_inductance(ej_ghz)
        C_r = 50e-15  # Resonator capacitance

        # Calculate qubit frequency (simplified)
        f_r = 1 / (2 * np.pi * np.sqrt(L * C))
        f_q_ghz = f_r / 1e9

        C_c = coupling_strength_to_capacitance(
            g_ghz=g_ghz,
            c_sigma=C,
            c_r=C_r,
            f_q_ghz=f_q_ghz,
            f_r_ghz=7.0,
        )

        # Create coupled model
        f = jnp.linspace(1e9, 20e9, 100)
        result = transmon_coupled(
            f=f,
            capacitance=C,
            inductance=L,
            coupling_capacitance=C_c,
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify all values are finite
        for value in result.values():
            assert jnp.isfinite(value).all()


@final
class TestQubitWithResonator(TwoPortModelTestSuite):
    """Tests for qubit_with_resonator model."""

    model_function = staticmethod(qubit_with_resonator)

    def test_with_converted_hamiltonian_parameters(self) -> None:
        """Test qubit_with_resonator with Hamiltonian parameters."""
        # Typical transmon parameters
        ec_ghz = 0.2  # Charging energy
        ej_ghz = 20.0  # Josephson energy

        C = ec_to_capacitance(ec_ghz)
        L = ej_to_inductance(ej_ghz)

        f = self.get_frequency_array(50)
        result = self._call_model(
            f=f,
            qubit_capacitance=C,
            qubit_inductance=L,
            resonator_length=5000.0,
            coupling_capacitance=5e-15,
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify all values are finite
        for value in result.values():
            assert jnp.isfinite(value).all()

    def test_grounded_qubit(self) -> None:
        """Test qubit_with_resonator with grounded qubit."""
        f = self.get_frequency_array(3)
        result = self._call_model(f=f, qubit_grounded=True)

        assert isinstance(result, dict)
        expected_keys = {("o1", "o1")}
        assert set(result.keys()) == expected_keys

    def test_different_resonator_lengths(self) -> None:
        """Test that different resonator lengths affect S-parameters."""
        f = self.get_frequency_array(50)

        result_short = self._call_model(f=f, resonator_length=3000.0)
        result_long = self._call_model(f=f, resonator_length=6000.0)

        # S-parameters should differ for different resonator lengths
        s11_short = result_short["o1", "o1"]
        s11_long = result_long["o1", "o1"]

        assert not jnp.allclose(s11_short, s11_long, atol=1e-3), (
            "Different resonator lengths should yield different S-parameters"
        )


class TestQubitWithResonatorWrappers:
    """Tests for qubit-resonator wrapper functions."""

    @staticmethod
    def test_flipmon_with_resonator() -> None:
        """Test flipmon_with_resonator returns valid S-params."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = flipmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_double_island_transmon_with_resonator() -> None:
        """Test double_island_transmon_with_resonator returns valid S-params."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = double_island_transmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_transmon_with_resonator_wrapper() -> None:
        """Test transmon_with_resonator wrapper returns valid S-params."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = transmon_with_resonator(f=f)
        assert isinstance(result, dict)
        for value in result.values():
            assert jnp.all(jnp.isfinite(value))

    @staticmethod
    def test_transmon_with_resonator_grounded() -> None:
        """Test transmon_with_resonator with grounded qubit."""
        f = jnp.linspace(4e9, 8e9, 10)
        result = transmon_with_resonator(f=f, qubit_grounded=True)
        assert isinstance(result, dict)
        # Grounded produces 1 port (o1 only)
        expected_keys = {("o1", "o1")}
        assert set(result.keys()) == expected_keys
