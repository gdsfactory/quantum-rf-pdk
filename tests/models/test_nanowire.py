"""Tests for qpdk.models.nanowire."""

import inspect
from typing import final, get_args, override

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from qpdk.cells.nanowire import nanowire_resonator as nanowire_resonator_cell
from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import π
from qpdk.models.nanowire import (
    grounded_pad_capacitance_analytical,
    nanowire_pad_capacitances_analytical,
    nanowire_resonator,
    nanowire_resonator_parameters,
)
from qpdk.models.waveguides import straight
from qpdk.tech import MeanderCornerStyle

from .base import TwoPortModelTestSuite

CORNER_STYLES = get_args(MeanderCornerStyle)
MEANDER: dict = {"n_turns": 5, "wire_width": 0.2, "wire_gap": 1.0}


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
    """A straight wire counts length/width plus its two leads, L_k = L_sheet x squares."""
    p = nanowire_resonator_parameters(
        wire_width=0.1, wire_length=40.0, sheet_inductance=40e-12
    )
    assert p["n_squares"] == pytest.approx(402.0)
    assert_allclose(p["kinetic_inductance"], 40e-12 * 402, rtol=1e-12)
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


@pytest.mark.parametrize("n_turns", [0, 2])
def test_parameters_reject_even_turns(n_turns: int) -> None:
    """The model accepts the same odd turn counts as the cell."""
    with pytest.raises(ValueError, match="n_turns"):
        nanowire_resonator_parameters(n_turns=n_turns)


# Method-of-moments extraction of the drawn 3D pads (pad, ground pad, side
# grounds and a 10/6 µm feedline on silicon, zero-thickness metal), graded mesh
# down to 2 µm. Columns: pad_width, pad_height, pad_separation, pad_ground_gap,
# coupling_gap, pad-to-ground C (fF), coupling C (fF).
MOM_REFERENCE = [
    (300.0, 150.0, 40.2, 50.0, 4.0, 76.70, 27.79),
    (300.0, 150.0, 40.2, 50.0, 10.0, 76.01, 20.51),
    (300.0, 80.0, 40.2, 30.0, 4.0, 56.98, 26.43),
    (200.0, 150.0, 60.0, 50.0, 8.0, 56.00, 15.48),
    (300.0, 150.0, 40.2, 20.0, 4.0, 83.02, 26.81),
]


@pytest.mark.parametrize("row", MOM_REFERENCE)
def test_pad_capacitances_match_method_of_moments(
    row: tuple[float, float, float, float, float, float, float],
) -> None:
    """The closed forms stay within the error bands stated in the docstring."""
    width, height, separation, ground_gap, gap, c_ground_ref, c_coupling_ref = row
    c_ground, c_coupling = nanowire_pad_capacitances_analytical(
        width, height, separation, ground_gap, gap, 10.0, 6.0, 11.45
    )
    assert 0.0 < float(c_ground) * 1e15 / c_ground_ref - 1 < 0.07
    assert -0.10 < float(c_coupling) * 1e15 / c_coupling_ref - 1 < 0.02


def test_coupling_capacitance_follows_layout() -> None:
    """C_c falls with coupling_gap and the override replaces it."""
    near = nanowire_resonator_parameters(coupling_gap=2.0)["coupling_capacitance"]
    far = nanowire_resonator_parameters(coupling_gap=20.0)["coupling_capacitance"]
    assert near > far > 0
    p = nanowire_resonator_parameters(coupling_capacitance=3e-15)
    assert p["coupling_capacitance"] == pytest.approx(3e-15)


def test_side_ground_adds_capacitance() -> None:
    """A narrower pad_ground_gap raises C and lowers f_r."""
    wide = nanowire_resonator_parameters(pad_ground_gap=100.0)
    narrow = nanowire_resonator_parameters(pad_ground_gap=10.0)
    assert narrow["capacitance"] > wide["capacitance"]
    assert narrow["resonance_frequency"] < wide["resonance_frequency"]


def test_feedline_phase_at_ports() -> None:
    """Far from resonance S21 is the drawn feedline's transmission."""
    f = jnp.array([20e9])
    length = 800.0
    s21 = nanowire_resonator(f=f, feedline_length=length, coupling_capacitance=1e-20)[
        "o2", "o1"
    ]
    ref = straight(f=f, length=length)["o2", "o1"]
    assert_allclose(s21, ref, atol=1e-6)
    assert jnp.abs(jnp.angle(ref[0])) > 0.1


@pytest.mark.parametrize("corner_style", CORNER_STYLES)
def test_jit_and_grad_over_geometry(corner_style: MeanderCornerStyle) -> None:
    """Wire length, pad height and wire width are traceable for every corner style."""
    f = jnp.array([1.5e9])

    def s21(x: jax.Array) -> jax.Array:
        wire_length, pad_height, wire_width = x
        s = nanowire_resonator(
            f=f,
            wire_length=wire_length,
            pad_height=pad_height,
            wire_width=wire_width,
            n_turns=MEANDER["n_turns"],
            wire_gap=MEANDER["wire_gap"],
            corner_style=corner_style,
        )
        return jnp.abs(s["o2", "o1"][0])

    def f_r(x: jax.Array) -> jax.Array:
        wire_length, pad_height, wire_width = x
        p = nanowire_resonator_parameters(
            wire_length=wire_length,
            pad_height=pad_height,
            wire_width=wire_width,
            n_turns=MEANDER["n_turns"],
            wire_gap=MEANDER["wire_gap"],
            corner_style=corner_style,
        )
        return jnp.asarray(p["resonance_frequency"])

    x = jnp.array([40.0, 150.0, 0.2])
    assert_allclose(jax.jit(s21)(x), s21(x), rtol=1e-10)
    grad_s21 = jax.jit(jax.grad(s21))(x)
    assert jnp.all(jnp.isfinite(grad_s21))
    # Longer, taller or narrower wire lowers f_r.
    grad_f_r = jax.jit(jax.grad(f_r))(x)
    assert grad_f_r[0] < 0
    assert grad_f_r[1] < 0
    assert grad_f_r[2] > 0
    eps = 1e-4
    for i in range(3):
        dx = jnp.zeros(3).at[i].set(eps * x[i])
        fd = (f_r(x + dx) - f_r(x - dx)) / (2 * dx[i])
        assert_allclose(grad_f_r[i], fd, rtol=1e-4)


def test_model_settings_match_cell() -> None:
    """Shared parameters have the same names and defaults in model and cell."""
    model = inspect.signature(nanowire_resonator).parameters
    cell = inspect.signature(nanowire_resonator_cell).parameters
    circuit_only = {"f", "sheet_inductance", "thickness", "coupling_capacitance"}
    shared = set(model) - circuit_only
    assert shared <= set(cell), shared - set(cell)
    for name in shared:
        assert model[name].default == cell[name].default, name


@pytest.mark.parametrize("corner_style", CORNER_STYLES)
def test_model_squares_match_cell(corner_style: MeanderCornerStyle) -> None:
    """The model counts the same squares the cell reports for its drawn wire."""
    cell = nanowire_resonator_cell(**MEANDER, corner_style=corner_style)
    model = nanowire_resonator_parameters(**MEANDER, corner_style=corner_style)
    assert_allclose(model["n_squares"], cell.info["n_squares"], rtol=1e-12)
