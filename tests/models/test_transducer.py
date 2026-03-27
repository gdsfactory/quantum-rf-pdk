"""Tests for transducer models."""

from __future__ import annotations

import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

from qpdk.models.transducer import (
    electro_optic_transducer,
    piezo_optomechanical_transducer,
    transducer_added_noise,
    transducer_cooperativity,
    transduction_bandwidth,
    transduction_efficiency,
)

# ---------------------------------------------------------------------------
# Transduction efficiency tests
# ---------------------------------------------------------------------------


class TestTransductionEfficiency:
    """Tests for the transduction_efficiency function."""

    def test_impedance_matched(self) -> None:
        """At C=1 and perfect extraction, η = 1."""
        eta = transduction_efficiency(1.0, eta_ext_mw=1.0, eta_ext_opt=1.0)
        assert jnp.isclose(eta, 1.0, atol=1e-6)

    def test_zero_cooperativity(self) -> None:
        """At C=0, efficiency must be zero."""
        eta = transduction_efficiency(0.0, eta_ext_mw=0.8, eta_ext_opt=0.8)
        assert jnp.isclose(eta, 0.0, atol=1e-10)

    def test_high_cooperativity(self) -> None:
        """At very high C, efficiency drops (over-coupling)."""
        eta_low = transduction_efficiency(1.0, eta_ext_mw=0.5, eta_ext_opt=0.5)
        eta_high = transduction_efficiency(100.0, eta_ext_mw=0.5, eta_ext_opt=0.5)
        assert float(eta_low) > float(eta_high)

    def test_symmetry(self) -> None:
        """Swapping MW and optical extraction efficiencies gives same result."""
        eta1 = transduction_efficiency(2.0, eta_ext_mw=0.3, eta_ext_opt=0.7)
        eta2 = transduction_efficiency(2.0, eta_ext_mw=0.7, eta_ext_opt=0.3)
        assert jnp.isclose(eta1, eta2, atol=1e-10)

    def test_bounded(self) -> None:
        """Efficiency must be in [0, 1]."""
        C_values = jnp.logspace(-3, 3, 100)
        for c_val in C_values:
            eta = transduction_efficiency(float(c_val), eta_ext_mw=1.0, eta_ext_opt=1.0)
            assert 0.0 <= float(eta) <= 1.0 + 1e-10

    @given(
        cooperativity=st.floats(min_value=0.01, max_value=100.0),
        eta_mw=st.floats(min_value=0.01, max_value=1.0),
        eta_opt=st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(deadline=None)
    def test_efficiency_bounded_hypothesis(
        self, cooperativity: float, eta_mw: float, eta_opt: float
    ) -> None:
        """Efficiency always in [0, 1] for valid inputs."""
        eta = transduction_efficiency(
            cooperativity, eta_ext_mw=eta_mw, eta_ext_opt=eta_opt
        )
        assert -1e-10 <= float(eta) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Cooperativity tests
# ---------------------------------------------------------------------------


class TestTransducerCooperativity:
    """Tests for the transducer_cooperativity function."""

    def test_basic_value(self) -> None:
        """C = 4g^2 / (κ_m κ_o) for known values."""
        g = 1e6  # 1 MHz
        kappa_mw = 10e6  # 10 MHz
        kappa_opt = 100e6  # 100 MHz
        expected = 4 * g**2 / (kappa_mw * kappa_opt)
        result = transducer_cooperativity(g, kappa_mw, kappa_opt)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_scaling_with_coupling(self) -> None:
        """Cooperativity scales as g^2."""
        c1 = transducer_cooperativity(1e6, 10e6, 100e6)
        c2 = transducer_cooperativity(2e6, 10e6, 100e6)
        assert jnp.isclose(c2 / c1, 4.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Added noise tests
# ---------------------------------------------------------------------------


class TestTransducerAddedNoise:
    """Tests for the transducer_added_noise function."""

    def test_zero_thermal(self) -> None:
        """With no thermal photons, added noise comes only from imperfect conversion."""
        n_add = transducer_added_noise(
            cooperativity=1.0,
            n_thermal_mw=0.0,
            n_thermal_opt=0.0,
            eta_ext_mw=1.0,
            eta_ext_opt=1.0,
        )
        # At C=1 with perfect coupling, the imperfect term is 0
        assert float(n_add) >= 0.0

    def test_monotonic_with_thermal(self) -> None:
        """More thermal photons → more noise."""
        n1 = transducer_added_noise(cooperativity=1.0, n_thermal_mw=0.01)
        n2 = transducer_added_noise(cooperativity=1.0, n_thermal_mw=0.1)
        assert float(n2) > float(n1)


# ---------------------------------------------------------------------------
# Bandwidth tests
# ---------------------------------------------------------------------------


class TestTransductionBandwidth:
    """Tests for the transduction_bandwidth function."""

    def test_basic_value(self) -> None:
        """Bandwidth = (κ_m + κ_o)/2 × (1 + C)."""
        bw = transduction_bandwidth(kappa_mw=10e6, kappa_opt=100e6, cooperativity=1.0)
        expected = (10e6 + 100e6) / 2.0 * 2.0
        assert jnp.isclose(bw, expected, rtol=1e-6)

    def test_broadens_with_cooperativity(self) -> None:
        """Higher cooperativity broadens the bandwidth."""
        bw1 = transduction_bandwidth(10e6, 100e6, 1.0)
        bw2 = transduction_bandwidth(10e6, 100e6, 10.0)
        assert float(bw2) > float(bw1)


# ---------------------------------------------------------------------------
# S-parameter model tests
# ---------------------------------------------------------------------------


class TestElectroOpticTransducer:
    """Tests for the electro_optic_transducer S-parameter model."""

    def test_default_parameters(self) -> None:
        """Model runs with default parameters."""
        S = electro_optic_transducer()
        assert ("o1", "o1") in S
        assert ("o1", "o2") in S

    def test_returns_stype(self) -> None:
        """Output is a valid SAX S-dictionary."""
        f = jnp.linspace(1e9, 10e9, 11)
        S = electro_optic_transducer(f=f)
        assert isinstance(S, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in S)

    def test_output_shape(self) -> None:
        """Output arrays have the same shape as input frequency."""
        f = jnp.linspace(1e9, 10e9, 51)
        S = electro_optic_transducer(f=f)
        assert S["o1", "o1"].shape == f.shape

    def test_passivity(self) -> None:
        """S-parameters must satisfy |S11|^2 + |S21|^2 <= 1."""
        f = jnp.linspace(1e9, 10e9, 101)
        S = electro_optic_transducer(f=f)
        power_sum = jnp.abs(S["o1", "o1"]) ** 2 + jnp.abs(S["o1", "o2"]) ** 2
        assert jnp.all(power_sum <= 1.0 + 1e-6)


class TestPiezoOptomechanicalTransducer:
    """Tests for the piezo_optomechanical_transducer S-parameter model."""

    def test_default_parameters(self) -> None:
        """Model runs with default parameters."""
        S = piezo_optomechanical_transducer()
        assert ("o1", "o1") in S
        assert ("o1", "o2") in S

    def test_returns_stype(self) -> None:
        """Output is a valid SAX S-dictionary."""
        f = jnp.linspace(1e9, 10e9, 11)
        S = piezo_optomechanical_transducer(f=f)
        assert isinstance(S, dict)

    def test_output_shape(self) -> None:
        """Output arrays have the same shape as input frequency."""
        f = jnp.linspace(1e9, 10e9, 51)
        S = piezo_optomechanical_transducer(f=f)
        assert S["o1", "o1"].shape == f.shape

    def test_passivity(self) -> None:
        """S-parameters must satisfy |S11|^2 + |S21|^2 <= 1."""
        f = jnp.linspace(1e9, 10e9, 101)
        S = piezo_optomechanical_transducer(f=f)
        power_sum = jnp.abs(S["o1", "o1"]) ** 2 + jnp.abs(S["o1", "o2"]) ** 2
        assert jnp.all(power_sum <= 1.0 + 1e-6)
