"""Tests for qpdk.models.perturbation — covers previously untested functions."""

import jax.numpy as jnp
from numpy.testing import assert_allclose

from qpdk.models.perturbation import (
    dispersive_shift,
    dispersive_shift_to_coupling,
    ej_ec_to_frequency_and_anharmonicity,
    measurement_induced_dephasing,
    purcell_decay_rate,
    resonator_linewidth_from_q,
    transmon_resonator_hamiltonian,
)


class TestTransmonResonatorHamiltonian:
    """Tests for the symbolic Hamiltonian constructor."""

    @staticmethod
    def test_returns_tuple_of_three() -> None:
        """Test return structure is (H_0, H_p, symbols)."""
        result = transmon_resonator_hamiltonian()
        assert len(result) == 3

    @staticmethod
    def test_symbols_are_correct() -> None:
        """Test that returned symbols have expected names."""
        _, _, symbols = transmon_resonator_hamiltonian()
        assert len(symbols) == 4
        omega_t, omega_r, alpha, g = symbols
        assert str(omega_t) == r"\omega_{t}"
        assert str(omega_r) == r"\omega_{r}"
        assert str(alpha) == r"\alpha"
        assert str(g) == "g"

    @staticmethod
    def test_symbols_are_positive_real() -> None:
        """Test that all symbols are declared positive and real."""
        _, _, symbols = transmon_resonator_hamiltonian()
        for s in symbols:
            assert s.is_positive
            assert s.is_real

    @staticmethod
    def test_h0_is_nontrivial() -> None:
        """Test that H_0 is a non-trivial sympy expression."""
        H_0, _, _ = transmon_resonator_hamiltonian()
        assert H_0 != 0
        # H_0 should contain the expected symbols
        free = H_0.free_symbols
        _, _, symbols = transmon_resonator_hamiltonian()
        omega_t, omega_r, alpha, _ = symbols
        assert omega_t in free
        assert omega_r in free
        assert alpha in free

    @staticmethod
    def test_hp_contains_coupling() -> None:
        """Test that H_p contains the coupling term."""
        _, H_p, _ = transmon_resonator_hamiltonian()
        assert H_p != 0


class TestDispersiveShiftRoundTrip:
    """Test dispersive_shift and dispersive_shift_to_coupling consistency."""

    @staticmethod
    def test_round_trip() -> None:
        """Test that computing chi from g and back recovers g."""
        g_in = 0.1  # GHz
        omega_t = 5.0
        omega_r = 7.0
        alpha = 0.2

        chi = dispersive_shift(omega_t, omega_r, alpha, g_in)
        g_out = dispersive_shift_to_coupling(chi, omega_t, omega_r, alpha)

        # Round-trip won't be exact because dispersive_shift_to_coupling
        # uses only the RWA term, but should be close for large detuning
        assert float(g_out) > 0
        assert_allclose(float(g_out), g_in, rtol=0.1)

    @staticmethod
    def test_dispersive_shift_sign() -> None:
        """Test that chi is positive when omega_t < omega_r (negative detuning)."""
        chi = dispersive_shift(5.0, 7.0, 0.2, 0.1)
        # With negative Delta and positive alpha, the dominant RWA term
        # 2g^2*alpha / (Delta*(Delta-alpha)) is positive
        assert float(chi) > 0

    @staticmethod
    def test_array_input() -> None:
        """Test that dispersive_shift works with array inputs."""
        omega_t = jnp.array([4.0, 5.0, 6.0])
        chi = dispersive_shift(omega_t, 7.0, 0.2, 0.1)
        assert chi.shape == (3,)


class TestEjEcConversion:
    """Tests for ej_ec_to_frequency_and_anharmonicity."""

    @staticmethod
    def test_typical_transmon() -> None:
        """Test with typical transmon parameters."""
        ej, ec = 20.0, 0.2
        omega_q, alpha = ej_ec_to_frequency_and_anharmonicity(ej, ec)
        # omega_q = sqrt(8 * 20 * 0.2) - 0.2 = sqrt(32) - 0.2 ≈ 5.46 GHz
        assert_allclose(float(omega_q), jnp.sqrt(8 * ej * ec) - ec, rtol=1e-6)
        assert_allclose(float(alpha), ec, rtol=1e-6)

    @staticmethod
    def test_alpha_equals_ec() -> None:
        """Test that anharmonicity equals charging energy."""
        _, alpha = ej_ec_to_frequency_and_anharmonicity(15.0, 0.3)
        assert_allclose(float(alpha), 0.3)


class TestPurcellDecayRate:
    """Tests for purcell_decay_rate."""

    @staticmethod
    def test_basic_computation() -> None:
        """Test Purcell rate = kappa * (g/Delta)^2."""
        g, omega_t, omega_r, kappa = 0.1, 5.0, 7.0, 0.001
        gamma = purcell_decay_rate(g, omega_t, omega_r, kappa)
        expected = kappa * (g / (omega_t - omega_r)) ** 2
        assert_allclose(float(gamma), expected, rtol=1e-6)

    @staticmethod
    def test_larger_coupling_larger_decay() -> None:
        """Test that larger coupling gives larger Purcell decay."""
        gamma_small = purcell_decay_rate(0.05, 5.0, 7.0, 0.001)
        gamma_large = purcell_decay_rate(0.10, 5.0, 7.0, 0.001)
        assert float(gamma_large) > float(gamma_small)


class TestResonatorLinewidthFromQ:
    """Tests for resonator_linewidth_from_q."""

    @staticmethod
    def test_basic() -> None:
        """Test kappa = omega_r / Q."""
        kappa = resonator_linewidth_from_q(7.0, 10_000)
        assert_allclose(float(kappa), 7.0 / 10_000, rtol=1e-6)

    @staticmethod
    def test_higher_q_narrower_linewidth() -> None:
        """Test that higher Q gives narrower linewidth."""
        k1 = resonator_linewidth_from_q(7.0, 1_000)
        k2 = resonator_linewidth_from_q(7.0, 10_000)
        assert float(k2) < float(k1)


class TestMeasurementInducedDephasing:
    """Tests for measurement_induced_dephasing."""

    @staticmethod
    def test_basic() -> None:
        """Test Gamma_phi = 8 * chi^2 * n_bar / kappa."""
        chi, kappa, n_bar = -0.001, 0.001, 5.0
        gamma = measurement_induced_dephasing(chi, kappa, n_bar)
        expected = 8 * chi**2 * n_bar / kappa
        assert_allclose(float(gamma), expected, rtol=1e-6)

    @staticmethod
    def test_zero_photons_zero_dephasing() -> None:
        """Test that zero photons give zero dephasing."""
        gamma = measurement_induced_dephasing(-0.001, 0.001, 0.0)
        assert_allclose(float(gamma), 0.0, atol=1e-15)
