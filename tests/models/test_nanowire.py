"""Tests for qpdk.models.nanowire."""

from typing import final, override

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import π
from qpdk.models.nanowire import (
    grounded_pad_capacitance_analytical,
    nanowire_resonator,
    nanowire_resonator_parameters,
)

from .base import TwoPortModelTestSuite


@final
class TestNanowireResonator(TwoPortModelTestSuite):
    """Passivity, reciprocity and shape checks for nanowire_resonator."""

    model_function = staticmethod(nanowire_resonator)

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        return {"n_turns": 3, "corner_style": "round", "wire_width": 0.2}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_turns": 5, "wire_width": 0.5, "corner_style": "optimal"},
        {"wire_width": 0.05, "wire_length": 10.0, "coupling_capacitance": 5e-15},
    ],
)
def test_resonance_matches_analytic_lc(kwargs: dict) -> None:
    """The |S21| dip sits at 1/(2π√(L(C + C_c))) and its width gives Q_c."""
    p = nanowire_resonator_parameters(**kwargs)
    c_total = p["capacitance"] + p["coupling_capacitance"]
    f_lc = 1 / (2 * π * jnp.sqrt(p["inductance"] * c_total))
    assert_allclose(p["resonance_frequency"], f_lc, rtol=1e-12)

    span = 5 / p["coupling_quality_factor"]
    f = jnp.linspace(f_lc * (1 - span), f_lc * (1 + span), 40001)
    s21 = jnp.abs(nanowire_resonator(f=f, **kwargs)["o1", "o2"])
    assert_allclose(f[jnp.argmin(s21)], f_lc, rtol=float(span) / 10000)

    # Lossless notch: 1 - |S21|^2 is a Lorentzian with FWHM f_r / Q_c.
    dip = f[1 - s21**2 > 0.5]
    assert_allclose(f_lc / (dip[-1] - dip[0]), p["coupling_quality_factor"], rtol=1e-2)


def test_kinetic_inductance_uses_drawn_squares() -> None:
    """A straight wire has length/width squares and L_k = L_sheet x squares."""
    p = nanowire_resonator_parameters(
        wire_width=0.1, wire_length=40.0, sheet_inductance=40e-12
    )
    assert p["n_squares"] == pytest.approx(400.0)
    assert_allclose(p["kinetic_inductance"], 40e-12 * 400, rtol=1e-12)
    assert 0 < p["geometric_inductance"] < 0.01 * p["kinetic_inductance"]


def test_grounded_pad_exceeds_symmetric_pads() -> None:
    """A half-plane ground gives more capacitance than an equal second pad."""
    grounded = grounded_pad_capacitance_analytical(300.0, 150.0, 40.0, 11.45)
    symmetric = plate_capacitor_capacitance_analytical(300.0, 150.0, 40.0, 11.45)
    assert grounded > symmetric


def test_grounded_pad_reduces_to_cross_ratio_modulus() -> None:
    """The symmetric pads with the same cross ratio give the same capacitance."""
    w, s = 150.0, 40.0
    # Symmetric pads of width w_eq and gap s_eq with the half-plane's cross ratio
    sqrt_lambda = jnp.sqrt(w / (w + s))
    k = (1 - sqrt_lambda) / (1 + sqrt_lambda)
    s_eq = 10.0
    w_eq = s_eq * (1 - k) / (2 * k)
    assert_allclose(
        grounded_pad_capacitance_analytical(300.0, w, s, 11.45),
        plate_capacitor_capacitance_analytical(300.0, w_eq, s_eq, 11.45),
        rtol=1e-6,
    )
