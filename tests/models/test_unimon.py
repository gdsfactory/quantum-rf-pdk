"""Tests for qpdk.models.unimon module - Unimon qubit models."""

from typing import ClassVar, final

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

import qpdk
import qpdk.models.unimon as unimon_module
from qpdk.models.constants import Φ_0, h
from qpdk.models.unimon import (
    el_to_arm_inductance,
    unimon_coupled,
    unimon_energies,
    unimon_frequency_and_anharmonicity,
    unimon_hamiltonian,
)
from qpdk.models.waveguides import straight_shorted

from .base import OnePortModelTestSuite

# Ensure PDK is activated for tests that require cross-section lookups
qpdk.PDK.activate()

MAX_EXAMPLES = 20


class TestElToArmInductance:
    """Tests for el_to_arm_inductance helper function."""

    @staticmethod
    def test_typical_value() -> None:
        """Test with typical unimon inductive energy."""
        el_ghz = 5.0
        L = el_to_arm_inductance(el_ghz)

        # L = Φ_0² / (8π² E_L)
        expected_L = Φ_0**2 / (8 * jnp.pi**2 * el_ghz * 1e9 * h)
        assert jnp.isclose(L, expected_L, rtol=1e-10)
        # Typical unimon inductance is ~10-20 nH
        assert 1e-9 < L < 100e-9, f"Inductance {L * 1e9:.2f} nH out of range"

    @staticmethod
    def test_inverse_relationship() -> None:
        """Test that inductance decreases as E_L increases."""
        L_low = el_to_arm_inductance(2.0)
        L_high = el_to_arm_inductance(10.0)
        assert L_low > L_high

    @staticmethod
    @given(el_ghz=st.floats(min_value=0.5, max_value=50.0))
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_inductance(el_ghz: float) -> None:
        """Test that inductance is always positive."""
        L = el_to_arm_inductance(el_ghz)
        assert L > 0


class TestUnimonHamiltonian:
    """Tests for the unimon Hamiltonian construction."""

    @staticmethod
    def test_hamiltonian_hermitian() -> None:
        """Test that the Hamiltonian is Hermitian."""
        H = unimon_hamiltonian(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_max=10)
        assert_allclose(H, H.T, atol=1e-12, err_msg="Hamiltonian must be Hermitian")

    @staticmethod
    def test_hamiltonian_shape() -> None:
        """Test that the Hamiltonian has correct shape."""
        n_max = 15
        H = unimon_hamiltonian(n_max=n_max)
        expected_size = 2 * n_max + 1
        assert H.shape == (expected_size, expected_size)

    @staticmethod
    def test_hamiltonian_real() -> None:
        """Test that the Hamiltonian is real for the unimon (no complex terms)."""
        H = unimon_hamiltonian(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0)
        assert jnp.allclose(jnp.imag(H), 0, atol=1e-12)

    @staticmethod
    @given(
        ec_ghz=st.floats(min_value=0.1, max_value=1.0),
        el_ghz=st.floats(min_value=1.0, max_value=50.0),
        ej_ghz=st.floats(min_value=1.0, max_value=50.0),
        phi_ext=st.floats(min_value=-2.0, max_value=2.0),
        n_max=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_hamiltonian_always_hermitian(
        ec_ghz: float, el_ghz: float, ej_ghz: float, phi_ext: float, n_max: int
    ) -> None:
        """Test Hermiticity with random parameters."""
        H = unimon_hamiltonian(
            ec_ghz=ec_ghz, el_ghz=el_ghz, ej_ghz=ej_ghz, phi_ext=phi_ext, n_max=n_max
        )
        assert_allclose(H, H.T, atol=1e-10)


class TestUnimonEnergies:
    """Tests for unimon energy spectrum computation."""

    @staticmethod
    def test_ground_state_zero() -> None:
        """Test that ground state energy is zero (after shifting)."""
        energies = unimon_energies(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_levels=3)
        assert_allclose(energies[0], 0.0, atol=1e-10)

    @staticmethod
    def test_energies_positive() -> None:
        """Test that all excited-state energies are positive."""
        energies = unimon_energies(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_levels=5)
        for i in range(1, len(energies)):
            assert energies[i] > 0, f"Energy level {i} should be positive"

    @staticmethod
    def test_energies_ordered() -> None:
        """Test that energy levels are in ascending order."""
        energies = unimon_energies(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_levels=5)
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"Energy levels should be ordered: E_{i} < E_{i + 1}"
            )

    @staticmethod
    def test_positive_anharmonicity_at_pi() -> None:
        """Test that anharmonicity is positive at phi_ext = pi.

        The unimon at its optimal operating point (phi_ext = pi) should
        have positive anharmonicity, which is one of its key advantages.
        """
        f01, alpha = unimon_frequency_and_anharmonicity(
            ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, phi_ext=jnp.pi
        )
        assert f01 > 0, "Qubit frequency should be positive"
        assert alpha > 0, "Anharmonicity should be positive at phi_ext = pi"

    @staticmethod
    def test_harmonic_limit() -> None:
        """Test that E_J = 0 gives the LC oscillator ladder sqrt(8 E_C E_L) * k."""
        energies = unimon_energies(ec_ghz=1.0, el_ghz=5.0, ej_ghz=0.0, n_levels=4)
        assert_allclose(energies, jnp.sqrt(8 * 1.0 * 5.0) * jnp.arange(4), atol=1e-9)

    @staticmethod
    def test_matches_scqubits_fluxonium() -> None:
        """Test against the scqubits fluxonium, which has the same single-mode Hamiltonian."""
        scq = pytest.importorskip("scqubits")
        for ec, el, ej, phi_ext in [(1.0, 5.0, 10.0, jnp.pi), (0.3, 2.0, 8.0, 1.0)]:
            fluxonium = scq.Fluxonium(
                EJ=ej, EC=ec, EL=el, flux=-phi_ext / (2 * jnp.pi), cutoff=150
            )
            expected = fluxonium.eigenvals(4)
            energies = unimon_energies(
                ec_ghz=ec, el_ghz=el, ej_ghz=ej, phi_ext=phi_ext, n_levels=4
            )
            assert_allclose(energies, expected - expected[0], atol=1e-6)

    @staticmethod
    def test_convergence_with_n_max() -> None:
        """Test that energies converge as n_max increases."""
        energies_small = unimon_energies(
            ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_max=15, n_levels=3
        )
        energies_large = unimon_energies(
            ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0, n_max=30, n_levels=3
        )
        # Low-lying energies should be close for sufficient truncation
        assert_allclose(energies_small, energies_large, rtol=1e-3)

    @staticmethod
    @given(
        ec_ghz=st.floats(min_value=0.1, max_value=1.0),
        el_ghz=st.floats(min_value=1.0, max_value=50.0),
        ej_ghz=st.floats(min_value=1.0, max_value=50.0),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_positive_transition_frequency(
        ec_ghz: float, el_ghz: float, ej_ghz: float
    ) -> None:
        """Test that the qubit transition frequency is always positive."""
        energies = unimon_energies(
            ec_ghz=ec_ghz,
            el_ghz=el_ghz,
            ej_ghz=ej_ghz,
            phi_ext=jnp.pi,
            n_max=20,
            n_levels=2,
        )
        assert energies[1] > 0


class TestUnimonFrequencyAndAnharmonicity:
    """Tests for the frequency and anharmonicity convenience function."""

    @staticmethod
    def test_returns_tuple() -> None:
        """Test that the function returns a tuple of two floats."""
        result = unimon_frequency_and_anharmonicity(ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        f01, alpha = result
        assert isinstance(f01, float)
        assert isinstance(alpha, float)

    @staticmethod
    def test_frequency_in_ghz_range() -> None:
        """Test that qubit frequency is in a physically reasonable range."""
        f01, _alpha = unimon_frequency_and_anharmonicity(
            ec_ghz=1.0, el_ghz=5.0, ej_ghz=10.0
        )
        # Unimon frequency should be in the microwave range (1-30 GHz)
        assert 1.0 < f01 < 30.0, f"Qubit frequency {f01:.2f} GHz out of expected range"


@final
class TestUnimonCoupledSAX(OnePortModelTestSuite):
    """Tests for the unimon coupled SAX S-parameter model."""

    model_function = staticmethod(unimon_coupled)
    expected_ports: ClassVar[set[str]] = {"coupling_o3"}

    def test_coupling_affects_response(self) -> None:
        """Test that coupling capacitance changes S-parameters."""
        f = self.get_frequency_array(50)
        result_weak = self._call_model(f=f, coupling_capacitance=1e-15)
        result_strong = self._call_model(f=f, coupling_capacitance=50e-15)
        s11_weak = result_weak["coupling_o3", "coupling_o3"]
        s11_strong = result_strong["coupling_o3", "coupling_o3"]
        assert not jnp.allclose(s11_weak, s11_strong, atol=1e-3), (
            "Different coupling capacitances should produce different S-parameters"
        )


@pytest.mark.parametrize(
    ("junction_inductance", "coupling_capacitance"),
    [(-7e-9, 10e-15), (9e-9, 20e-15)],
)
def test_internal_node_has_zero_injected_current(
    junction_inductance: float, coupling_capacitance: float
) -> None:
    f = jnp.linspace(1.4e9, 11.9e9, 211)
    omega = 2 * jnp.pi * f
    arm_reflection = straight_shorted(f=f, length=3000)["o1", "o1"]
    arm = (1 - arm_reflection) / (50 * (1 + arm_reflection))
    junction = 1j * omega * 5e-15 + 1 / (1j * omega * junction_inductance)
    qubit = arm + junction - junction**2 / (arm + junction)
    coupling = 1j * omega * coupling_capacitance
    admittance = coupling * qubit / (coupling + qubit)
    expected = (1 - 50 * admittance) / (1 + 50 * admittance)
    actual = jax.jit(unimon_coupled)(
        f=f,
        junction_inductance=junction_inductance,
        coupling_capacitance=coupling_capacitance,
    )["coupling_o3", "coupling_o3"]
    assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


def test_closed_unimon_with_lossless_arms_conserves_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lossless_arm(f: jax.Array, length: float, **_settings: str) -> dict:
        reflection = -jnp.exp(-4j * jnp.pi * f * length * 1e-6 / 1.2e8)
        return {("o1", "o1"): reflection}

    monkeypatch.setattr(unimon_module, "straight_shorted", lossless_arm)
    model = jax.jit(lambda f: unimon_coupled(f=f))
    response = model(jnp.linspace(1e9, 12e9, 2001))["coupling_o3", "coupling_o3"]
    assert_allclose(jnp.abs(response), 1.0, atol=1e-11)
