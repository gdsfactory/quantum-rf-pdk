"""Tests for qpdk.models.perturbation module."""

import math

import hypothesis.strategies as st
from hypothesis import given, settings

from qpdk.models.perturbation import (
    dispersive_shift,
    dispersive_shift_to_coupling,
    ej_ec_to_frequency_and_anharmonicity,
    measurement_induced_dephasing,
    purcell_decay_rate,
    resonator_linewidth_from_q,
)

MAX_EXAMPLES = 20


class TestEjEcToFrequencyAndAnharmonicity:
    """Tests for :func:`~ej_ec_to_frequency_and_anharmonicity`."""

    @staticmethod
    def test_returns_tuple() -> None:
        omega_q, alpha = ej_ec_to_frequency_and_anharmonicity(20.0, 0.2)
        expected_f_q = math.sqrt(8 * 20.0 * 0.2) - 0.2
        assert math.isclose(omega_q, expected_f_q, rel_tol=1e-10)
        assert math.isclose(alpha, 0.2)

    @staticmethod
    def test_frequency_increases_with_ej() -> None:
        """Test that frequency increases with E_J."""
        f_low, _ = ej_ec_to_frequency_and_anharmonicity(10.0, 0.2)
        f_high, _ = ej_ec_to_frequency_and_anharmonicity(40.0, 0.2)
        assert f_high > f_low


class TestDispersiveShift:
    """Tests for :func:`~dispersive_shift`."""

    @staticmethod
    def test_sign_for_negative_detuning() -> None:
        """Test sign convention for ω_t < ω_r (typical case)."""
        chi = dispersive_shift(5.0, 7.0, 0.2, 0.1)
        # For negative detuning (ω_t < ω_r), χ should be positive
        # since the dominant term has Δ < 0 and α > 0
        assert chi > 0

    @staticmethod
    def test_quadratic_in_g() -> None:
        """Dispersive shift should scale as g²."""
        chi1 = dispersive_shift(5.0, 7.0, 0.2, 0.1)
        chi2 = dispersive_shift(5.0, 7.0, 0.2, 0.2)
        # χ ∝ g², so chi2/chi1 ≈ 4
        ratio = chi2 / chi1
        assert math.isclose(ratio, 4.0, rel_tol=1e-10)

    @staticmethod
    def test_zero_coupling() -> None:
        """Zero coupling should give zero dispersive shift."""
        chi = dispersive_shift(5.0, 7.0, 0.2, 0.0)
        assert chi == 0.0

    @given(
        g=st.floats(min_value=0.01, max_value=0.3),
        omega_t=st.floats(min_value=3.0, max_value=7.0),
        omega_r=st.floats(min_value=5.0, max_value=10.0),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_finite_result(self, g: float, omega_t: float, omega_r: float) -> None:
        """Result should be finite for non-degenerate parameters."""
        # Avoid resonance (omega_t = omega_r) and omega_t + omega_r = alpha
        if abs(omega_t - omega_r) < 0.1 or abs(omega_t - omega_r - 0.2) < 0.1:
            return
        chi = dispersive_shift(omega_t, omega_r, 0.2, g)
        assert math.isfinite(chi)


class TestDispersiveShiftToCoupling:
    """Tests for :func:`~dispersive_shift_to_coupling`."""

    @staticmethod
    def test_round_trip() -> None:
        """Converting g→χ→g should return the original g."""
        omega_t, omega_r, alpha = 5.0, 7.0, 0.2
        g_original = 0.08

        # Compute χ from g using only the dominant RWA term
        # (matching the inversion formula used in dispersive_shift_to_coupling)
        delta = omega_t - omega_r
        chi_rwa = 2 * g_original**2 * alpha / (delta * (delta - alpha))

        g_recovered = dispersive_shift_to_coupling(chi_rwa, omega_t, omega_r, alpha)
        assert math.isclose(g_recovered, g_original, rel_tol=1e-8)

    @staticmethod
    def test_positive_result() -> None:
        """Coupling should always be positive."""
        g = dispersive_shift_to_coupling(-0.001, 5.0, 7.0, 0.2)
        assert g > 0


class TestPurcellDecayRate:
    """Tests for :func:`purcell_decay_rate`."""

    @staticmethod
    def test_typical_value() -> None:
        gamma = purcell_decay_rate(0.1, 5.0, 7.0, 0.001)
        # gamma = kappa * (g/Delta)^2 = 0.001 * (0.1/2)^2 = 0.0025e-3
        expected = 0.001 * (0.1 / 2.0) ** 2
        assert math.isclose(gamma, expected, rel_tol=1e-10)

    @staticmethod
    def test_zero_coupling() -> None:
        gamma = purcell_decay_rate(0.0, 5.0, 7.0, 0.001)
        assert gamma == 0.0


class TestResonatorLinewidthFromQ:
    """Tests for :func:`~resonator_linewidth_from_q`."""

    @staticmethod
    def test_typical_value() -> None:
        kappa = resonator_linewidth_from_q(7.0, 10_000)
        assert math.isclose(kappa, 7.0 / 10_000)


class TestMeasurementInducedDephasing:
    """Tests for :func:`~measurement_induced_dephasing`."""

    @staticmethod
    def test_formula() -> None:
        gamma = measurement_induced_dephasing(-0.001, 0.001, 5.0)
        # 8 * 0.001^2 * 5 / 0.001 = 8 * 1e-6 * 5 / 1e-3 = 0.04
        expected = 8 * 0.001**2 * 5.0 / 0.001
        assert math.isclose(gamma, expected, rel_tol=1e-10)
