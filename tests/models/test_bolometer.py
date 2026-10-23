"""Tests for the normal-metal nanobolometer thermal and RF models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

from qpdk.cells import nanobolometer as nanobolometer_cell
from qpdk.models.bolometer import (
    AUPD_ELECTRON_PHONON_COUPLING,
    AUPD_HEAT_CAPACITY_COEFFICIENT,
    LORENZ_NUMBER,
    andreev_heat_suppression,
    electron_heat_capacity,
    electron_phonon_conductance,
    nanobolometer,
    nanobolometer_thermal_modes,
    nanocalorimeter,
    thermal_relaxation_modes,
    two_reservoir_relaxation,
    wiedemann_franz_conductance,
)
from qpdk.models.constants import e, k_B

positive = st.floats(min_value=1e-3, max_value=1e3)


def test_lorenz_number() -> None:
    """The Sommerfeld value is π² k_B² / (3 e²) ≈ 2.44e-8 W Ω K⁻²."""
    assert pytest.approx(2.443e-8, rel=1e-3) == LORENZ_NUMBER
    assert wiedemann_franz_conductance(1.0, 1.0) == pytest.approx(LORENZ_NUMBER)


def test_single_reservoir_time_constant() -> None:
    """τ = C / G_ep = γ / (5 Σ T³) is independent of volume."""
    volume, temperature = 1e-21, 0.02
    tau = electron_heat_capacity(
        volume, temperature, AUPD_HEAT_CAPACITY_COEFFICIENT
    ) / electron_phonon_conductance(volume, temperature, AUPD_ELECTRON_PHONON_COUPLING)
    expected = AUPD_HEAT_CAPACITY_COEFFICIENT / (
        5 * AUPD_ELECTRON_PHONON_COUPLING * temperature**3
    )
    assert tau == pytest.approx(expected)


def test_andreev_suppression_limits() -> None:
    """Heat flow is unsuppressed without a gap and exponentially blocked below it."""
    # the truncated Li2 series converges slowest at y = 0
    assert andreev_heat_suppression(0.1, 0.0) == pytest.approx(1.0, rel=1e-3)
    gap = 213e-6 * e
    deep = andreev_heat_suppression(0.02, gap)
    assert 0 < deep < 1e-40
    y = gap / (k_B * 0.1)
    # BRT asymptote for y >> 1: (2 y² + 4 y + 4) e^{-y} / (π²/6) to leading order
    assert andreev_heat_suppression(0.1, gap) == pytest.approx(
        (y**2 + 2 * y + 2) * jnp.exp(-y) / (jnp.pi**2 / 6), rel=0.05
    )


@settings(deadline=None)
@given(c1=positive, c2=positive, g1=positive, g2=positive, g12=positive)
def test_two_reservoir_matches_eigh(
    c1: float, c2: float, g1: float, g2: float, g12: float
) -> None:
    """The analytic 2x2 solution agrees with the general eigen-solver."""
    tau_1, tau_2, a_1, a_2 = two_reservoir_relaxation(c1, c2, g1, g2, g12)
    taus, amplitudes = thermal_relaxation_modes(
        jnp.array([c1, c2]), jnp.array([[g1 + g12, -g12], [-g12, g2 + g12]])
    )
    assert tau_1 <= tau_2
    assert jnp.allclose(jnp.array([tau_1, tau_2]), taus, rtol=1e-6)
    assert jnp.allclose(jnp.array([a_1, a_2]), amplitudes, rtol=1e-5, atol=1e-9)
    assert a_1 + a_2 == pytest.approx(1.0)
    assert 0 <= a_1 <= 1
    assert 0 <= a_2 <= 1


def test_decoupled_limit_is_single_exponential() -> None:
    """Without a link, reservoir 1 relaxes as one exponential with C1/G1."""
    tau_1, tau_2, a_1, a_2 = two_reservoir_relaxation(2.0, 5.0, 4.0, 1.0, 0.0)
    assert a_1 == pytest.approx(1.0)
    assert a_2 == pytest.approx(0.0, abs=1e-12)
    assert tau_1 == pytest.approx(0.5)
    assert tau_2 == pytest.approx(5.0)


def test_strong_link_limit() -> None:
    """A strong link gives τ_slow = (C1+C2)/(G1+G2) with a_slow/a_fast = C1/C2."""
    c1, c2, g1, g2 = 3.0, 1.5, 0.3, 0.2
    tau_1, tau_2, a_1, a_2 = two_reservoir_relaxation(c1, c2, g1, g2, 1e9)
    assert tau_2 == pytest.approx((c1 + c2) / (g1 + g2), rel=1e-6)
    assert a_2 / a_1 == pytest.approx(c1 / c2, rel=1e-6)
    assert tau_1 < 1e-8


def test_layout_single_reservoir_limit() -> None:
    """With no metal beyond the absorber edge the layout model collapses to γ/(5ΣT³)."""
    temperature = 0.03
    taus, amplitudes = nanobolometer_thermal_modes(
        contact_overlap=1e-6, extension_length=0.0, temperature=temperature
    )
    expected = AUPD_HEAT_CAPACITY_COEFFICIENT / (
        5 * AUPD_ELECTRON_PHONON_COUPLING * temperature**3
    )
    assert taus[1] == pytest.approx(expected, rel=1e-4)
    assert amplitudes[1] == pytest.approx(1.0, rel=1e-5)


def test_layout_material_reservoir() -> None:
    """A weakly coupled material reservoir adds a slow mode; off by default."""
    taus_2, amplitudes_2 = nanobolometer_thermal_modes()
    assert taus_2.shape == (2,)
    taus_3, amplitudes_3 = nanobolometer_thermal_modes(
        material_heat_capacity=1e-19, material_conductance=1e-16
    )
    assert taus_3.shape == (3,)
    assert jnp.all(jnp.diff(taus_3) > 0)
    assert jnp.sum(amplitudes_3) == pytest.approx(1.0)
    assert jnp.sum(amplitudes_2) == pytest.approx(1.0)
    # vanishing coupling recovers the two-reservoir result
    taus_0, _ = nanobolometer_thermal_modes(
        material_heat_capacity=1e-19, material_conductance=0.0
    )
    assert jnp.allclose(jnp.sort(taus_0)[:2], taus_2, rtol=1e-6)


def test_layout_matches_cell_volumes() -> None:
    """The model's reservoir volumes match the cell's geometry-derived ``info``."""
    c = nanobolometer_cell()
    width, thickness = 0.135, 0.03
    assert c.info["absorber_volume"] == pytest.approx(1.0 * width * thickness)
    assert c.info["reservoir_volume"] == pytest.approx(2 * 0.5 * width * thickness)
    assert c.info["link_length"] == pytest.approx(0.25)


def test_layout_modes_are_jittable() -> None:
    """The geometry-to-modes path traces through ``jax.jit``."""
    taus, _ = jax.jit(nanobolometer_thermal_modes)(temperature=0.05)
    assert jnp.all(jnp.isfinite(taus))


@settings(deadline=None)
@given(
    length=st.floats(min_value=0.05, max_value=100.0),
    resistivity=st.floats(min_value=1e-9, max_value=1e-5),
)
def test_rf_model_is_passive(length: float, resistivity: float) -> None:
    """A resistive load reflects no more power than is incident."""
    s = nanobolometer(
        f=jnp.array([4e9, 8e9]), absorber_length=length, resistivity=resistivity
    )
    assert jnp.all(jnp.abs(s["o1", "o1"]) <= 1.0)


def test_rf_model_matched_load() -> None:
    """R = Z0 gives no reflection; the default 24.7 Ω absorber reflects -0.34."""
    width, thickness = 0.135, 0.03
    matched_length = 50.0 * width * thickness * 1e-6 / 1e-7
    s = nanobolometer(f=6e9, absorber_length=matched_length)
    assert jnp.abs(s["o1", "o1"]) == pytest.approx(0.0, abs=1e-9)
    s_default = nanobolometer(f=6e9)
    assert jnp.real(s_default["o1", "o1"]) == pytest.approx(-0.339, abs=1e-3)


def test_rf_model_calorimeter_is_cu_wire() -> None:
    """The calorimeter model is a 360 Ω Cu wire, the resistance of the reference device."""
    s = nanocalorimeter(f=6e9)
    assert jnp.real(s["o1", "o1"]) == pytest.approx((360.0 - 50.0) / 410.0, rel=1e-6)


# Geometry and material inputs from the papers alone, without per-device tuning.
# Gunyhó et al., Nat. Electron. 9, 611 (2026), arXiv:2412.14079, Methods Eq. 8:
# τ1 ≈ 18 µs, τ2 ≈ 150 µs, a2/a1 ≈ 1.5. All uncovered AuPd (1 µm absorber plus
# 8 × 0.3 µm SNS junctions) is lumped into the absorber and the Al-covered metal
# (≈ 10 × 0.3 µm) into the second reservoir.
GUNYHO_2026 = {
    "absorber_length": 3.4,
    "absorber_width": 0.135,
    "normal_metal_thickness": 0.03,
    "contact_overlap": 1.5,
    "extension_length": 0.0,
}
# Gasparinetti et al., PRApplied 3, 014007 (2015), arXiv:1405.7568, Suppl. Fig. 6:
# τ1 = 97 µs, τ2 = 0.41 ms, A2/A1 = 0.018 at T_e ≈ 85 mK. Cu, γ = 71, Σ = 2e9.
GASPARINETTI_2015 = {
    "absorber_length": 20.0,
    "absorber_width": 0.1,
    "normal_metal_thickness": 0.025,
    "contact_overlap": 0.2,
    "extension_length": 0.2,
    "heat_capacity_coefficient": 71.0,
    "electron_phonon_coupling": 2e9,
    "resistivity": 4.5e-8,
}


def test_reference_device_gunyho() -> None:
    """Pin the geometric prediction for the Aalto nanobolometer.

    The amplitude ratio a_slow/a_fast = C1/C2 ≈ 1.13 is the right order of the
    measured 1.5, but the Wiedemann-Franz link equilibrates the two reservoirs
    in ~1 ns instead of the measured 18 µs. The slow mode is the plain
    electron-phonon time γ/(5ΣT³), which matches 150 µs only near 52 mK.
    """
    taus, amplitudes = nanobolometer_thermal_modes(**GUNYHO_2026, temperature=0.05)
    assert taus[0] == pytest.approx(1.42e-9, rel=0.01)
    assert taus[1] == pytest.approx(309e-6, rel=0.01)
    assert amplitudes[1] / amplitudes[0] == pytest.approx(1.133, rel=1e-3)
    # the measured fast mode is four orders of magnitude slower than predicted
    assert taus[0] < 18e-6 / 1e3


def test_reference_device_gasparinetti() -> None:
    """Pin the geometric prediction for the Pekola Cu calorimeter.

    The main mode is the electron-phonon time, about 8x faster than the
    measured 97 µs (the paper's own heat-capacity excess). The layout reservoir
    adds a ps mode of weight 0.04 rather than a slow 0.018 tail.
    """
    taus, amplitudes = nanobolometer_thermal_modes(
        **GASPARINETTI_2015, temperature=0.085
    )
    assert taus[0] == pytest.approx(10.1e-12, rel=0.01)
    assert taus[1] == pytest.approx(11.6e-6, rel=0.01)
    assert amplitudes[0] == pytest.approx(0.0385, rel=0.01)
    # even the dominant electron-phonon mode is faster than the measured τ1
    assert taus[1] < 97e-6
