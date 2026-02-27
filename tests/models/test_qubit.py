"""Tests for qpdk.models.qubit module - Qubit LC resonator models."""

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import scipy.constants
from hypothesis import given, settings

from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    double_island_transmon,
    ec_to_capacitance,
    ej_to_inductance,
    qubit_with_resonator,
    shunted_transmon,
    transmon_coupled,
)

MAX_EXAMPLES = 20

# Physical constants for reference
_e = scipy.constants.e
_h = scipy.constants.h
_Φ_0 = scipy.constants.physical_constants["mag. flux quantum"][0]


class TestEcToCapacitance:
    """Tests for ec_to_capacitance helper function."""

    def test_typical_value(self) -> None:
        """Test with typical transmon charging energy."""
        # E_C = 0.2 GHz is typical for transmons
        ec_ghz = 0.2
        C = ec_to_capacitance(ec_ghz)

        # C_Σ = e² / (2 * E_C)
        # For E_C = 0.2 GHz, C_Σ ≈ 96 fF
        expected_C = _e**2 / (2 * ec_ghz * 1e9 * _h)
        assert np.isclose(C, expected_C, rtol=1e-10)
        # Check reasonable range
        assert 50e-15 < C < 200e-15, f"Capacitance {C * 1e15:.1f} fF out of range"

    def test_inverse_relationship(self) -> None:
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

    def test_typical_value(self) -> None:
        """Test with typical transmon Josephson energy."""
        # E_J = 20 GHz is typical for transmons
        ej_ghz = 20.0
        L = ej_to_inductance(ej_ghz)

        # L_J = Φ_0² / (4π² E_J)
        expected_L = _Φ_0**2 / (4 * np.pi**2 * ej_ghz * 1e9 * _h)
        assert np.isclose(L, expected_L, rtol=1e-10)
        # Check reasonable range (~ 1 nH for 20 GHz)
        assert 0.1e-9 < L < 10e-9, f"Inductance {L * 1e9:.2f} nH out of range"

    def test_inverse_relationship(self) -> None:
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

    def test_typical_value(self) -> None:
        """Test with typical qubit-resonator coupling parameters."""
        # Typical values:
        # g = 0.1 GHz, C_Σ = 80 fF, C_r = 50 fF, ω_q = 5 GHz, ω_r = 7 GHz
        C_c = coupling_strength_to_capacitance(
            g_ghz=0.1,
            c_sigma=80e-15,
            c_r=50e-15,
            omega_q_ghz=5.0,
            omega_r_ghz=7.0,
        )

        # Should give a few fF coupling capacitance
        assert 0.1e-15 < C_c < 50e-15, (
            f"Coupling capacitance {C_c * 1e15:.2f} fF out of range"
        )

    def test_coupling_increases_with_g(self) -> None:
        """Test that coupling capacitance increases with g."""
        C_c_weak = coupling_strength_to_capacitance(
            g_ghz=0.05,
            c_sigma=80e-15,
            c_r=50e-15,
            omega_q_ghz=5.0,
            omega_r_ghz=7.0,
        )
        C_c_strong = coupling_strength_to_capacitance(
            g_ghz=0.2,
            c_sigma=80e-15,
            c_r=50e-15,
            omega_q_ghz=5.0,
            omega_r_ghz=7.0,
        )
        assert C_c_strong > C_c_weak

    @given(
        g_ghz=st.floats(min_value=0.01, max_value=0.5),
        c_sigma=st.floats(min_value=10e-15, max_value=200e-15),
        c_r=st.floats(min_value=10e-15, max_value=200e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_coupling_capacitance(
        self, g_ghz: float, c_sigma: float, c_r: float
    ) -> None:
        """Test that coupling capacitance is always positive."""
        C_c = coupling_strength_to_capacitance(
            g_ghz=g_ghz,
            c_sigma=c_sigma,
            c_r=c_r,
            omega_q_ghz=5.0,
            omega_r_ghz=7.0,
        )
        assert C_c > 0


class TestDoubleIslandTransmon:
    """Tests for double_island_transmon model."""

    def test_default_parameters(self) -> None:
        """Test double_island_transmon with default parameters."""
        result = double_island_transmon()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_returns_stype(self) -> None:
        """Test that double_island_transmon returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = double_island_transmon(f=f, capacitance=100e-15, inductance=1e-9)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys

    def test_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = double_island_transmon(f=f)

        for value in result.values():
            assert len(value) == n_freq

    def test_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 20e9, 50)
        result = double_island_transmon(f=f)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10

    def test_passivity(self) -> None:
        """Test that the model satisfies passivity (energy conservation)."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = double_island_transmon(f=f)

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_is_ungrounded(self) -> None:
        """Test that double-island transmon has ungrounded behavior."""
        f = jnp.array([5e9, 10e9, 15e9])
        result = double_island_transmon(f=f)

        # For ungrounded, S22 should NOT be -1 (which would indicate a short)
        s22 = result[("o2", "o2")]
        assert not jnp.allclose(s22, -1.0, atol=1e-6), (
            "Double-island transmon should NOT be grounded"
        )

        # For an ungrounded parallel LC, S22 should equal S11 due to symmetry
        s11 = result[("o1", "o1")]
        assert jnp.allclose(s11, s22, atol=1e-10), (
            "S11 and S22 should be equal for symmetric ungrounded LC resonator"
        )

    def test_resonance_frequency(self) -> None:
        """Test that the resonance frequency matches expected f_r = 1/(2*pi*sqrt(LC))."""
        L = 1e-9  # 1 nH
        C = 100e-15  # 100 fF
        f_r_expected = 1 / (2 * np.pi * np.sqrt(L * C))

        # Fine frequency sweep around resonance
        f = jnp.linspace(f_r_expected * 0.8, f_r_expected * 1.2, 1000)
        result = double_island_transmon(f=f, capacitance=C, inductance=L)

        # At resonance, |S21| should be minimum (parallel LC has infinite impedance)
        s21 = result[("o2", "o1")]
        s21_mag = jnp.abs(s21)
        idx_min = jnp.argmin(s21_mag)
        f_observed = f[idx_min]

        relative_error = abs(float(f_observed - f_r_expected) / f_r_expected)
        assert relative_error < 0.01


class TestShuntedTransmon:
    """Tests for shunted_transmon model."""

    def test_default_parameters(self) -> None:
        """Test shunted_transmon with default parameters."""
        result = shunted_transmon()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_returns_stype(self) -> None:
        """Test that shunted_transmon returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = shunted_transmon(f=f, capacitance=100e-15, inductance=1e-9)

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys

    def test_is_grounded(self) -> None:
        """Test that shunted transmon has grounded behavior."""
        f = jnp.array([5e9, 10e9, 15e9])
        result = shunted_transmon(f=f)

        # For grounded, S22 should be -1 (short reflection)
        s22 = result[("o2", "o2")]
        assert jnp.allclose(s22, -1.0, atol=1e-10), (
            f"S22 should be -1 for grounded port, got {s22}"
        )

        # S21 should be zero (no transmission through ground)
        s21 = result[("o2", "o1")]
        assert jnp.allclose(s21, 0.0, atol=1e-10)

    def test_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = shunted_transmon(f=f)

        for value in result.values():
            assert len(value) == n_freq

    @given(
        L=st.floats(min_value=0.1e-9, max_value=10e-9),
        C=st.floats(min_value=10e-15, max_value=1000e-15),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_with_hypothesis(self, L: float, C: float) -> None:
        """Test shunted transmon with random valid L and C values."""
        f = jnp.linspace(1e9, 30e9, 50)
        result = shunted_transmon(f=f, capacitance=C, inductance=L)

        assert isinstance(result, dict)
        assert len(result) == 4


class TestTransmonCoupled:
    """Tests for transmon_coupled model."""

    def test_default_parameters(self) -> None:
        """Test transmon_coupled with default parameters."""
        result = transmon_coupled()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = transmon_coupled(f=f, coupling_capacitance=10e-15)

        for value in result.values():
            assert len(value) == n_freq

    def test_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 20e9, 50)
        result = transmon_coupled(f=f, coupling_capacitance=10e-15)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10

    def test_passivity(self) -> None:
        """Test that the coupled model satisfies passivity."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = transmon_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=1e-9
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_capacitive_coupling(self) -> None:
        """Test coupled transmon with capacitive coupling only."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = transmon_coupled(
            f=f, coupling_capacitance=10e-15, coupling_inductance=0.0
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify passivity
        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_inductive_coupling(self) -> None:
        """Test coupled transmon with inductive coupling only."""
        f = jnp.linspace(1e9, 30e9, 100)
        result = transmon_coupled(
            f=f, coupling_capacitance=0.0, coupling_inductance=1e-9
        )

        assert isinstance(result, dict)
        assert len(result) == 4

    def test_grounded_coupling(self) -> None:
        """Test coupled transmon with grounded qubit."""
        f = jnp.array([5e9, 10e9, 15e9])
        result = transmon_coupled(f=f, grounded=True, coupling_capacitance=10e-15)

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
        f = jnp.linspace(1e9, 30e9, 50)
        result = transmon_coupled(
            f=f,
            coupling_capacitance=coupling_C,
            coupling_inductance=coupling_L,
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        # Verify passivity
        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]
        total_power = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
        assert jnp.all(total_power <= 1.0 + 1e-6)


class TestIntegration:
    """Integration tests combining helper functions with qubit models."""

    def test_ec_ej_to_qubit_model(self) -> None:
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

    def test_coupled_qubit_with_converted_parameters(self) -> None:
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
        omega_q_ghz = f_r / 1e9

        C_c = coupling_strength_to_capacitance(
            g_ghz=g_ghz,
            c_sigma=C,
            c_r=C_r,
            omega_q_ghz=omega_q_ghz,
            omega_r_ghz=7.0,
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
            assert jnp.all(jnp.isfinite(value))


class TestQubitWithResonator:
    """Tests for qubit_with_resonator model."""

    def test_default_parameters(self) -> None:
        """Test qubit_with_resonator with default parameters."""
        import qpdk

        qpdk.PDK.activate()
        result = qubit_with_resonator()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert ("o1", "o1") in result, "Should have S11 parameter"
        assert ("o1", "o2") in result, "Should have S12 parameter"

    def test_returns_stype(self) -> None:
        """Test that qubit_with_resonator returns a valid sax.SType dictionary."""
        import qpdk

        qpdk.PDK.activate()
        f = jnp.array([5e9, 6e9, 7e9])
        result = qubit_with_resonator(
            f=f,
            qubit_capacitance=100e-15,
            qubit_inductance=1e-9,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys

    def test_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        import qpdk

        qpdk.PDK.activate()
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = qubit_with_resonator(f=f)

        for value in result.values():
            assert len(value) == n_freq

    def test_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        import qpdk

        qpdk.PDK.activate()
        f = jnp.linspace(3e9, 20e9, 50)
        result = qubit_with_resonator(f=f)

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-10

    def test_passivity(self) -> None:
        """Test that the model satisfies passivity (energy conservation)."""
        import qpdk

        qpdk.PDK.activate()
        f = jnp.linspace(1e9, 30e9, 100)
        result = qubit_with_resonator(f=f)

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_with_converted_hamiltonian_parameters(self) -> None:
        """Test qubit_with_resonator with Hamiltonian parameters."""
        import qpdk

        qpdk.PDK.activate()

        # Typical transmon parameters
        ec_ghz = 0.2  # Charging energy
        ej_ghz = 20.0  # Josephson energy

        C = ec_to_capacitance(ec_ghz)
        L = ej_to_inductance(ej_ghz)

        f = jnp.linspace(4e9, 8e9, 50)
        result = qubit_with_resonator(
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
            assert jnp.all(jnp.isfinite(value))

    def test_grounded_qubit(self) -> None:
        """Test qubit_with_resonator with grounded qubit."""
        import qpdk

        qpdk.PDK.activate()
        f = jnp.array([5e9, 10e9, 15e9])
        result = qubit_with_resonator(f=f, qubit_grounded=True)

        assert isinstance(result, dict)
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys

    def test_different_resonator_lengths(self) -> None:
        """Test that different resonator lengths affect S-parameters."""
        import qpdk

        qpdk.PDK.activate()
        f = jnp.linspace(4e9, 8e9, 50)

        result_short = qubit_with_resonator(f=f, resonator_length=3000.0)
        result_long = qubit_with_resonator(f=f, resonator_length=6000.0)

        # S-parameters should differ for different resonator lengths
        s11_short = result_short[("o1", "o1")]
        s11_long = result_long[("o1", "o1")]

        assert not jnp.allclose(s11_short, s11_long, atol=1e-3), (
            "Different resonator lengths should yield different S-parameters"
        )
