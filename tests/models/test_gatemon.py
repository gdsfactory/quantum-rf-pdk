"""Tests for qpdk.models.gatemon module - Gatemon qubit models."""

from itertools import pairwise
from typing import final

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

from qpdk.models.constants import e
from qpdk.models.gatemon import (
    andreev_bound_state_energies,
    gatemon,
    gatemon_energies,
    gatemon_frequency_and_anharmonicity,
    gatemon_hamiltonian,
    gatemon_josephson_energy,
    gatemon_potential,
    gatemon_supercurrent,
)
from qpdk.models.qubit import ej_to_inductance, shunted_transmon

from .base import TwoPortModelTestSuite

MAX_EXAMPLES = 20
EC_GHZ = 0.26  # Device A of Riechert et al. (2025)


def _cosine_transmon(
    ec_ghz: float, ej_ghz: float, n_g: float = 0.0, n_max: int = 30
) -> tuple[float, float]:
    """Return f01 and alpha of a tunnel-junction transmon in the charge basis."""
    n = np.arange(-n_max, n_max + 1)
    hopping = np.eye(n.size, k=1) + np.eye(n.size, k=-1)
    H = np.diag(4 * ec_ghz * (n - n_g) ** 2) - ej_ghz / 2 * hopping
    energies = np.linalg.eigvalsh(H)
    energies -= energies[0]
    return energies[1], energies[2] - 2 * energies[1]


def _transmon_ej_for_frequency(ec_ghz: float, f01_ghz: float) -> float:
    """Return the transmon E_J giving the qubit frequency ``f01_ghz`` (bisection)."""
    low, high = 0.1, 200.0
    for _ in range(100):
        mid = (low + high) / 2
        if _cosine_transmon(ec_ghz, mid)[0] < f01_ghz:
            low = mid
        else:
            high = mid
    return (low + high) / 2


class TestJunction:
    """Tests for the Andreev bound state junction potential and current."""

    @staticmethod
    def test_abs_energy_range() -> None:
        """Test that E_p goes from Δ at φ=0 to Δ sqrt(1-τ) at φ=π."""
        energies = andreev_bound_state_energies(
            jnp.array([0.0, jnp.pi]), 50.0, (0.36, 1.0)
        )
        assert energies.shape == (2, 2)
        assert_allclose(energies[0], [50.0, 50.0])
        assert_allclose(energies[1], [40.0, 0.0], atol=1e-6)

    @staticmethod
    def test_tunnel_limit_is_cosine() -> None:
        """Test that U → -E_J cos φ + const with E_J = Δ Σ τ / 4 as τ → 0."""
        delta_ghz, transmissions = 4e4, (5e-4, 5e-4)
        ej_ghz = float(gatemon_josephson_energy(delta_ghz, transmissions))
        assert_allclose(ej_ghz, 10.0)

        phi = jnp.linspace(-jnp.pi, jnp.pi, 101)
        potential = gatemon_potential(phi, delta_ghz, transmissions)
        offset = -delta_ghz * len(transmissions) + ej_ghz
        assert_allclose(potential - offset, -ej_ghz * jnp.cos(phi), atol=5e-3)

    @staticmethod
    def test_josephson_energy_is_curvature() -> None:
        """Test that E_J equals d²U/dφ² at φ = 0 for any transmission."""
        delta_ghz, transmissions = 50.0, (0.9, 0.3)
        curvature = jax.grad(jax.grad(gatemon_potential))(0.0, delta_ghz, transmissions)
        assert_allclose(
            curvature, gatemon_josephson_energy(delta_ghz, transmissions), rtol=1e-10
        )
        assert_allclose(curvature, 50.0 * 1.2 / 4, rtol=1e-10)

    @staticmethod
    def test_supercurrent_tunnel_limit() -> None:
        """Test that I(φ) → (2e/ħ) E_J sin φ in the tunnel limit."""
        phi = jnp.linspace(0, jnp.pi, 9)
        current = gatemon_supercurrent(phi, 4e4, (1e-3,))
        # (2e/ħ) × h × 1e9 × E_J[GHz] with E_J = 10 GHz
        expected = 4 * jnp.pi * e * 1e9 * 10.0 * jnp.sin(phi)
        assert_allclose(current, expected, rtol=1e-3, atol=1e-15)

    @staticmethod
    def test_supercurrent_ballistic_channel() -> None:
        """Test that one ballistic channel carries a critical current eΔ/ħ."""
        delta_ghz = 50.0
        phi = jnp.linspace(0, jnp.pi - 1e-4, 2001)
        current = gatemon_supercurrent(phi, delta_ghz, (1.0,))
        # eΔ/ħ with Δ = h × 1e9 × Δ[GHz]
        assert_allclose(jnp.max(current), 2 * jnp.pi * e * 1e9 * delta_ghz, rtol=1e-6)

    @staticmethod
    def test_supercurrent_drops_with_temperature() -> None:
        """Test that thermal occupation reduces the supercurrent."""
        cold = gatemon_supercurrent(jnp.pi / 2, 50.0, (0.8,), 0.0)
        warm = gatemon_supercurrent(jnp.pi / 2, 50.0, (0.8,), 2.0)
        assert 0 < warm < cold


class TestGatemonHamiltonian:
    """Tests for the gatemon charge-basis Hamiltonian."""

    @staticmethod
    def test_hamiltonian_shape_and_symmetry() -> None:
        """Test that H is a real symmetric (2 n_max + 1)-square matrix."""
        H = gatemon_hamiltonian(n_max=12)
        assert H.shape == (25, 25)
        assert_allclose(H, H.T, atol=1e-12)
        assert not jnp.iscomplexobj(H)

    @staticmethod
    def test_tunnel_limit_has_only_nearest_neighbour_coupling() -> None:
        """Test that only U_{±1} = -E_J/2 couples charge states for τ → 0."""
        H = gatemon_hamiltonian(EC_GHZ, 4e4, (1e-3,), n_max=5)
        assert_allclose(jnp.diag(H, 1), -5.0, rtol=1e-3)
        assert_allclose(jnp.diag(H, 2), 0.0, atol=2e-3)

    @staticmethod
    @given(
        ec_ghz=st.floats(min_value=0.1, max_value=1.0),
        delta_ghz=st.floats(min_value=10.0, max_value=200.0),
        tau=st.floats(min_value=0.0, max_value=1.0),
        n_g=st.floats(min_value=-0.5, max_value=0.5),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_hamiltonian_always_symmetric(
        ec_ghz: float, delta_ghz: float, tau: float, n_g: float
    ) -> None:
        """Test symmetry with random parameters."""
        H = gatemon_hamiltonian(ec_ghz, delta_ghz, (tau,), n_g, n_max=10)
        assert_allclose(H, H.T, atol=1e-10)


class TestGatemonSpectrum:
    """Tests for the gatemon qubit frequency and anharmonicity."""

    @staticmethod
    def test_energies_start_at_zero_and_increase() -> None:
        """Test that the levels are referenced to the ground state and sorted."""
        energies = gatemon_energies(n_levels=4)
        assert energies.shape == (4,)
        assert_allclose(energies[0], 0.0, atol=1e-12)
        assert jnp.all(jnp.diff(energies) > 0)

    @staticmethod
    def test_tunnel_limit_matches_transmon() -> None:
        """Test that τ → 0 reproduces the cosine transmon spectrum."""
        for ej_ghz in (2.0, 10.0):
            for n_g in (0.0, 0.25):
                f01, alpha = gatemon_frequency_and_anharmonicity(
                    EC_GHZ, 4 * ej_ghz / 1e-3, (1e-3,), n_g
                )
                f01_ref, alpha_ref = _cosine_transmon(EC_GHZ, ej_ghz, n_g)
                assert_allclose(f01, f01_ref, rtol=1e-3)
                assert_allclose(alpha, alpha_ref, rtol=2e-3)

    @staticmethod
    def test_default_spectrum_regression() -> None:
        """Pin the default spectrum (f01 ≈ 4.45 GHz, α ≈ -127 MHz).

        Regression values from this implementation, cross-checked against an
        independent phase-basis solver; not a literature value.
        """
        f01, alpha = gatemon_frequency_and_anharmonicity()
        assert_allclose(f01, 4.447, atol=5e-3)
        assert_allclose(alpha, -0.127, atol=2e-3)

    @staticmethod
    def test_high_transmission_reduces_anharmonicity() -> None:
        """Test that at fixed f01 a τ ≈ 1 channel has smaller |α| than an SIS transmon."""
        f01, alpha = gatemon_frequency_and_anharmonicity(EC_GHZ, 40.0 / 0.99, (0.99,))
        ej_transmon = _transmon_ej_for_frequency(EC_GHZ, float(f01))
        f01_transmon, alpha_transmon = _cosine_transmon(EC_GHZ, ej_transmon)
        assert_allclose(f01_transmon, f01, rtol=1e-6)

        assert alpha < 0
        assert abs(alpha) < abs(alpha_transmon)
        # Near τ = 1 a single channel approaches |α| ≈ E_C / 4
        assert abs(alpha) < EC_GHZ / 2

    @staticmethod
    def test_anharmonicity_decreases_with_transmission() -> None:
        """Test that |α| falls monotonically with τ at fixed E_J."""
        alphas = [
            abs(
                float(
                    gatemon_frequency_and_anharmonicity(EC_GHZ, 40.0 / tau, (tau,))[1]
                )
            )
            for tau in (0.1, 0.5, 0.9, 1.0)
        ]
        assert all(a > b for a, b in pairwise(alphas))

    @staticmethod
    def test_charge_dispersion_grows_at_low_ej_over_ec() -> None:
        """Test that f01(n_g=0) - f01(n_g=0.5) grows as E_J/E_C drops."""
        dispersions = []
        for ratio in (40, 20, 10, 5):
            delta_ghz = 4 * ratio * EC_GHZ / 0.8
            f_0 = gatemon_frequency_and_anharmonicity(EC_GHZ, delta_ghz, (0.8,), 0.0)[0]
            f_half = gatemon_frequency_and_anharmonicity(
                EC_GHZ, delta_ghz, (0.8,), 0.5
            )[0]
            dispersions.append(abs(float(f_0 - f_half)))
        assert all(a < b for a, b in pairwise(dispersions))
        assert dispersions[-1] > 0.1  # > 100 MHz at E_J/E_C = 5

    @staticmethod
    def test_jittable_and_vmappable() -> None:
        """Test that the spectrum compiles under jit and maps over the offset charge."""
        n_g = jnp.linspace(0, 0.5, 4)
        f01, alpha = jax.vmap(
            lambda ng: gatemon_frequency_and_anharmonicity(EC_GHZ, 50.0, (0.8,), ng)
        )(n_g)
        assert f01.shape == alpha.shape == (4,)
        f01_jit = jax.jit(
            lambda d: gatemon_frequency_and_anharmonicity(EC_GHZ, d, jnp.array([0.8]))[
                0
            ]
        )(50.0)
        assert_allclose(f01_jit, f01[0], rtol=1e-10)


@final
class TestGatemon(TwoPortModelTestSuite):
    """Tests for the gatemon SAX model."""

    model_function = staticmethod(gatemon)

    def test_matches_shunted_transmon(self) -> None:
        """Test that gatemon equals shunted_transmon with L_J from E_J = Δ Σ τ / 4."""
        f = self.get_frequency_array(50)
        result = self._call_model(f=f, delta_ghz=60.0, transmissions=(0.5, 0.3))
        expected = shunted_transmon(f=f, inductance=ej_to_inductance(12.0))
        for key, value in expected.items():
            assert_allclose(result[key], value, atol=1e-12)

    def test_default_josephson_energy(self) -> None:
        """Test that the defaults give E_J = 10 GHz."""
        f = self.get_frequency_array(20)
        expected = shunted_transmon(f=f, inductance=ej_to_inductance(10.0))
        for key, value in self._call_model(f=f).items():
            assert_allclose(value, expected[key], atol=1e-12)

    def test_resonance_tracks_transmission(self) -> None:
        """Test that quadrupling Σ τ doubles the LC resonance frequency."""
        C = 100e-15
        f_r = []
        for tau in (0.2, 0.8):
            L = float(ej_to_inductance(gatemon_josephson_energy(50.0, (tau,))))
            f_expected = 1 / (2 * jnp.pi * jnp.sqrt(L * C))
            f = jnp.linspace(f_expected * 0.8, f_expected * 1.2, 2001)
            s11 = self._call_model(f=f, capacitance=C, transmissions=(tau,))["o1", "o1"]
            f_r.append(float(f[jnp.argmin(jnp.abs(s11 - 1.0))]))
            assert abs(f_r[-1] - float(f_expected)) / float(f_expected) < 0.01
        assert_allclose(f_r[1] / f_r[0], 2.0, rtol=0.01)
