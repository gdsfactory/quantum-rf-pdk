"""Tests for qpdk.models.esr."""

from typing import final, override

import gdsfactory as gf
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose

from qpdk import PDK
from qpdk.models.constants import h, μ_0, π
from qpdk.models.esr import (
    esr_lc_resonator,
    esr_lc_resonator_parameters,
    esr_mode_volume,
    thin_strip_magnetic_field,
)
from qpdk.models.inductor import self_inductance_strip

from .base import TwoPortModelTestSuite

ħ = h / (2 * π)

# Design values: Bienfait et al. (2016), Ranjan et al. (2020).
BIENFAIT = {"wire_width": 5.0, "wire_length": 730.0, "f0": 7.26e9, "zc": 44.0}
RANJAN = {"wire_width": 0.1, "wire_length": 10.0, "f0": 7.25e9, "zc": 15.0}


@final
class TestEsrLcResonator(TwoPortModelTestSuite):
    """Passivity, reciprocity and shape checks for esr_lc_resonator."""

    model_function = staticmethod(esr_lc_resonator)

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        return {"wire_width": 0.5, "wire_length": 50.0, "fingers": 6}


def test_model_jits_over_geometry() -> None:
    """The model jits with the geometry and coupling as traced arguments."""

    @jax.jit
    def s21(wire_width: float, wire_length: float, cc: float, cg: float) -> jax.Array:
        return esr_lc_resonator(
            f=jnp.array([6e9, 7e9]),
            wire_width=wire_width,
            wire_length=wire_length,
            coupling_capacitance=cc,
            ground_capacitance=cg,
        )["o1", "o2"]

    jitted = s21(2.0, 300.0, 5e-15, 50e-15)
    eager = esr_lc_resonator(
        f=jnp.array([6e9, 7e9]),
        wire_width=2.0,
        wire_length=300.0,
        coupling_capacitance=5e-15,
        ground_capacitance=50e-15,
    )["o1", "o2"]
    assert_allclose(jitted, eager, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"wire_width": 0.1, "wire_length": 10.0, "fingers": 20},
        {"wire_width": 1.0, "wire_length": 200.0, "coupling_capacitance": 1e-14},
        {"ground_capacitance": 20e-15, "coupling_capacitance": 2e-14},
    ],
)
def test_resonance_and_coupling_q(kwargs: dict) -> None:
    """The |S21| dip sits at 1/(2π√(L C_eff)) and its width gives Q_c."""
    p = esr_lc_resonator_parameters(**kwargs)
    c_c, c_g = p["coupling_capacitance"], p["ground_capacitance"]
    # C in parallel with C_g in series with (C_g + C_c).
    c_eff = p["capacitance"] + 1 / (1 / c_g + 1 / (c_g + c_c))
    assert_allclose(p["effective_capacitance"], c_eff, rtol=1e-12)
    f_lc = 1 / (2 * π * jnp.sqrt(p["inductance"] * c_eff))
    assert_allclose(p["resonance_frequency"], f_lc, rtol=1e-12)

    span = 5 / p["coupling_quality_factor"]
    f = jnp.linspace(f_lc * (1 - span), f_lc * (1 + span), 40001)
    s21 = jnp.abs(esr_lc_resonator(f=f, **kwargs)["o1", "o2"])
    assert_allclose(f[jnp.argmin(s21)], f_lc, rtol=float(span) / 10000)

    # Lossless notch: 1 - |S21|^2 is a Lorentzian with FWHM f_r / Q_c.
    dip = f[1 - s21**2 > 0.5]
    assert_allclose(f_lc / (dip[-1] - dip[0]), p["coupling_quality_factor"], rtol=1e-2)


def test_floating_resonator_needs_ground_capacitance() -> None:
    """Without C_g the floating LC has no return path and leaves S21 flat."""
    p = esr_lc_resonator_parameters(ground_capacitance=0.0)
    assert jnp.isinf(p["coupling_quality_factor"])
    f = jnp.linspace(6e9, 8e9, 2001)
    s21 = jnp.abs(esr_lc_resonator(f=f, ground_capacitance=0.0)["o1", "o2"])
    assert_allclose(s21, 1.0, atol=1e-9)


def test_singular_limits_are_finite() -> None:
    """DC, C_c = 0 and the bare-LC frequency give finite S; the notch reflects."""
    p = esr_lc_resonator_parameters()
    f_bare = 1 / (2 * π * jnp.sqrt(p["inductance"] * p["capacitance"]))
    f = jnp.array([0.0, f_bare, p["resonance_frequency"]])
    s = esr_lc_resonator(f=f)
    assert_allclose(s["o1", "o2"][0], 1.0)
    assert_allclose(s["o1", "o1"][0], 0.0)
    assert jnp.all(jnp.isfinite(s["o1", "o2"]))
    assert_allclose(s["o1", "o1"][2], -1.0, atol=1e-6)
    assert_allclose(s["o1", "o2"][2], 0.0, atol=1e-6)

    s = esr_lc_resonator(f=f, coupling_capacitance=0.0)
    assert_allclose(s["o1", "o2"], 1.0)
    assert_allclose(s["o1", "o1"], 0.0)

    # The safe-value substitution also keeps the gradient finite at DC.
    grad = jax.grad(
        lambda c: jnp.sum(
            jnp.abs(esr_lc_resonator(f=f, coupling_capacitance=c)["o1", "o2"])
        )
    )(8.5e-15)
    assert jnp.isfinite(grad)


def test_bienfait_design_values() -> None:
    """The default wire reproduces the paper's L = Z_c/ω0 and f0 within 10 %."""
    p = esr_lc_resonator_parameters()
    assert p["wire_squares"] == pytest.approx(146.0)
    l_paper = BIENFAIT["zc"] / (2 * π * BIENFAIT["f0"])
    assert_allclose(p["inductance"], l_paper, rtol=0.1)
    assert_allclose(p["resonance_frequency"], BIENFAIT["f0"], rtol=0.1)


def test_ranjan_wire_squares() -> None:
    """The 100 nm x 10 µm wire has 100 squares."""
    p = esr_lc_resonator_parameters(
        wire_width=RANJAN["wire_width"], wire_length=RANJAN["wire_length"]
    )
    assert p["wire_squares"] == pytest.approx(100.0)


def test_default_q_c_matches_ranjan_qext() -> None:
    """The 8.5 fF default gives Q_c of order 3e4, Ranjan's Q_ext.

    The default is a chosen parameter, not the coupling of the default layout.
    """
    q_c = esr_lc_resonator_parameters()["coupling_quality_factor"]
    assert 2e4 < q_c < 4e4


@pytest.mark.parametrize(
    ("design", "volume_paper"),
    [(BIENFAIT, 0.02e6), (RANJAN, 6.0)],
    ids=["bienfait", "ranjan"],
)
def test_mode_volume_order_of_magnitude(design: dict, volume_paper: float) -> None:
    """With the paper's own L, V_m is within a factor 4 of the simulated value.

    The papers quote EM-simulated volumes (~0.02 nl, ~6 fL); the closed form is
    an estimate of the same order, not a reproduction. Z_c/ω0 includes
    inductance outside the wire's field, so this is a loose consistency check.
    """
    inductance = design["zc"] / (2 * π * design["f0"])
    volume = esr_mode_volume(design["wire_width"], inductance)
    assert volume_paper / 4 < volume < volume_paper * 4


def test_mode_volume_matches_single_photon_field() -> None:
    """V_m reproduces δB1 = √(μ0 ħ ω0 / 2 V_m) with δi = ω0 √(ħ / 2 Z_c)."""
    w, f0, zc = RANJAN["wire_width"], RANJAN["f0"], RANJAN["zc"]
    omega0 = 2 * π * f0
    delta_i = omega0 * jnp.sqrt(ħ / (2 * zc))
    delta_b1 = μ_0 * delta_i / (2 * w * 1e-6)
    volume = esr_mode_volume(w, zc / omega0) * 1e-18
    assert_allclose(delta_b1, jnp.sqrt(μ_0 * ħ * omega0 / (2 * volume)), rtol=1e-12)


@pytest.mark.parametrize(
    ("wire_width", "wire_length"), [(5.0, 730.0), (0.1, 10.0), (1.0, 120.0)]
)
def test_cell_info_matches_model(wire_width: float, wire_length: float) -> None:
    """The cell's mode_volume uses the wire's geometric inductance."""
    PDK.activate()
    c = gf.get_component(
        "esr_lc_resonator", wire_width=wire_width, wire_length=wire_length
    )
    inductance = self_inductance_strip(
        wire_length * 1e-6, wire_width * 1e-6, 0.05 * 1e-6
    )
    assert c.info["wire_squares"] == pytest.approx(wire_length / wire_width)
    assert_allclose(
        c.info["mode_volume"], esr_mode_volume(wire_width, inductance), rtol=1e-9
    )


def test_strip_field_limits() -> None:
    """Near the surface B_x → μ0 I/(2w); far away it is a line current."""
    current, width = 1e-3, 5.0
    b_x, b_y = thin_strip_magnetic_field(current, width, 1e-4)
    assert_allclose(b_x, -μ_0 * current / (2 * width * 1e-6), rtol=1e-4)
    assert_allclose(b_y, 0.0, atol=1e-15)

    r = 1e5
    b_x, _ = thin_strip_magnetic_field(current, width, r)
    assert_allclose(b_x, -μ_0 * current / (2 * π * r * 1e-6), rtol=1e-8)


def test_strip_field_matches_line_current_sum() -> None:
    """The closed form equals a numerical sum of line currents across the strip."""
    current, width, height, x = 2e-3, 3.0, 0.7, 1.1
    n = 20000
    xs = (jnp.arange(n) + 0.5) / n * width - width / 2
    dx, dy = (x - xs) * 1e-6, height * 1e-6
    r2 = dx**2 + dy**2
    di = current / n
    # Line current along +z: B = μ0 I/(2π r²) (-dy, dx).
    b_x_sum = jnp.sum(-μ_0 * di * dy / (2 * π * r2))
    b_y_sum = jnp.sum(μ_0 * di * dx / (2 * π * r2))
    b_x, b_y = thin_strip_magnetic_field(current, width, height, x)
    assert_allclose(b_x, b_x_sum, rtol=1e-6)
    assert_allclose(b_y, b_y_sum, rtol=1e-6)


@settings(deadline=None)
@given(
    width=st.floats(0.1, 5.0),
    height=st.floats(0.01, 100.0),
    x=st.floats(-10.0, 10.0),
)
def test_strip_field_symmetry(width: float, height: float, x: float) -> None:
    """B_x is even and B_y odd in x, and |B_x| above the centre is the maximum."""
    b_x, b_y = thin_strip_magnetic_field(1e-3, width, height, x)
    b_x_m, b_y_m = thin_strip_magnetic_field(1e-3, width, height, -x)
    b_x_0, _ = thin_strip_magnetic_field(1e-3, width, height)
    # B_y vanishes at x = 0, so compare it on the scale of the peak field.
    atol = 1e-9 * float(jnp.abs(b_x_0))
    assert_allclose(b_x, b_x_m, rtol=1e-9)
    assert_allclose(b_y, -b_y_m, rtol=1e-9, atol=atol)
    assert jnp.abs(b_x) <= jnp.abs(b_x_0) * (1 + 1e-12)
    # Mirror image below the strip: B_x reverses, B_y is unchanged.
    b_x_below, b_y_below = thin_strip_magnetic_field(1e-3, width, -height, x)
    assert_allclose(b_x_below, -b_x, rtol=1e-9)
    assert_allclose(b_y_below, b_y, rtol=1e-9, atol=atol)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wire_width": 0.05}, "wire_width"),
        ({"wire_width": 6.0}, "wire_width"),
        ({"fingers": 1}, "fingers"),
        ({"coupling_gap": 0.0}, "coupling_gap"),
        ({"wire_width": 5.0, "finger_length": 4.0}, "finger_length"),
        ({"wire_width": 5.0, "finger_thickness": 4.0}, "finger_thickness"),
        ({"wire_length": 851.0}, "span"),
        ({"feedline_length": 100.0}, "feedline_length"),
        ({"fingers": 7, "finger_gap": 50.001, "finger_thickness": 49.999}, "0.002"),
    ],
)
def test_cell_rejects_invalid_geometry(kwargs: dict, match: str) -> None:
    """The cell raises on out-of-range geometry."""
    PDK.activate()
    with pytest.raises(ValueError, match=match):
        gf.get_component("esr_lc_resonator", **kwargs)
