"""Tests for the giant-atom transmon models."""

from typing import final, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from qpdk.models.constants import c_0, π
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.generic import capacitor, tee
from qpdk.models.giant_atom import (
    giant_atom_decay_rate_and_lamb_shift,
    giant_atom_transmon,
)
from qpdk.models.qubit import shunted_transmon

from .base import TwoPortModelTestSuite

QUBIT_CAPACITANCE = 100e-15
QUBIT_INDUCTANCE = 7e-9


def _bare_frequency(coupling_capacitances: tuple[float, ...]) -> float:
    """LC frequency of the qubit loaded by its coupling capacitances."""
    c_sigma = QUBIT_CAPACITANCE + sum(coupling_capacitances)
    return 1 / (2 * np.pi * np.sqrt(QUBIT_INDUCTANCE * c_sigma))


def _fit_dip(
    coupling_capacitances: tuple[float, ...],
    segment_lengths: tuple[float, ...],
    centre: float,
    width: float,
) -> tuple[float, float]:
    """Return the FWHM and the centre of the dip in ``1 - |S21|^2`` in Hz."""
    f = jnp.linspace(centre - 20 * width, centre + 20 * width, 40001)
    s = giant_atom_transmon(
        f=f,
        coupling_capacitances=coupling_capacitances,
        segment_lengths=segment_lengths,
        qubit_capacitance=QUBIT_CAPACITANCE,
        qubit_inductance=QUBIT_INDUCTANCE,
    )
    loss = 1 - np.abs(np.asarray(s["o1", "o2"])) ** 2
    peak = int(loss.argmax())
    above = np.flatnonzero(loss >= loss[peak] / 2)
    f = np.asarray(f)
    return float(f[above[-1]] - f[above[0]]), float(f[peak])


@final
class TestGiantAtomTransmon(TwoPortModelTestSuite):
    """Generic S-parameter checks: shape, reciprocity and passivity."""

    model_function = giant_atom_transmon


@final
class TestGiantAtomTransmonThreePoints(TwoPortModelTestSuite):
    """Generic S-parameter checks for three coupling points."""

    model_function = giant_atom_transmon

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        return {
            "coupling_capacitances": (5e-15, 8e-15, 3e-15),
            "segment_lengths": (9000.0, 4000.0),
        }


def test_jit() -> None:
    """The model and the analytic helper compile with jax.jit."""
    f = jnp.linspace(5e9, 6e9, 11)
    s = jax.jit(giant_atom_transmon)(f)
    assert all(jnp.all(jnp.isfinite(v)) for v in s.values())
    rates = jax.jit(giant_atom_decay_rate_and_lamb_shift)(f)
    assert all(jnp.all(jnp.isfinite(r)) for r in rates)


def test_single_point_is_small_atom() -> None:
    """With K = 1 the netlist equals a capacitively side-coupled transmon."""
    f = jnp.linspace(5.0e9, 6.5e9, 201)
    cc = 8e-15
    giant = giant_atom_transmon(
        f=f,
        coupling_capacitances=(cc,),
        segment_lengths=(),
        qubit_capacitance=QUBIT_CAPACITANCE,
        qubit_inductance=QUBIT_INDUCTANCE,
    )
    small = sax.evaluate_circuit_fg(
        (
            {"line,o3": "coupler,o1", "coupler,o2": "qubit,o1"},
            {"o1": "line,o1", "o2": "line,o2"},
        ),
        {
            "line": tee(f=f),
            "coupler": capacitor(f=f, capacitance=cc),
            "qubit": shunted_transmon(
                f=f, capacitance=QUBIT_CAPACITANCE, inductance=QUBIT_INDUCTANCE
            ),
        },
    )
    for key in [("o1", "o1"), ("o1", "o2"), ("o2", "o2")]:
        np.testing.assert_allclose(giant[key], small[key], rtol=1e-6, atol=1e-9)


def test_single_point_rates() -> None:
    """With K = 1 the analytic rate is γ and there is no Lamb shift."""
    cc = 8e-15
    f0 = _bare_frequency((cc,))
    gamma, delta = giant_atom_decay_rate_and_lamb_shift(
        f0, (cc,), (), QUBIT_CAPACITANCE
    )
    c_sigma = QUBIT_CAPACITANCE + cc
    expected = (2 * np.pi * f0) ** 2 * cc**2 * 50.0 / (2 * c_sigma)
    np.testing.assert_allclose(gamma, expected, rtol=1e-12)
    assert abs(float(delta)) < 1e-12 * float(gamma)


@pytest.mark.parametrize(
    ("coupling_capacitances", "segment_lengths"),
    [
        ((8e-15,), ()),
        ((8e-15, 8e-15), (13100.0,)),
        ((8e-15, 8e-15), (6552.0,)),
        ((8e-15, 8e-15, 8e-15), (20000.0, 20000.0)),
    ],
    ids=["small_atom", "two_points_lamb_down", "two_points_lamb_up", "three_points"],
)
def test_analytic_matches_sax_linewidth(
    coupling_capacitances: tuple[float, ...], segment_lengths: tuple[float, ...]
) -> None:
    """Analytic Γ and Δ match the SAX linewidth and shift for weak coupling."""
    f0 = _bare_frequency(coupling_capacitances)
    gamma, delta = giant_atom_decay_rate_and_lamb_shift(
        f0, coupling_capacitances, segment_lengths, QUBIT_CAPACITANCE
    )
    gamma_hz = float(gamma) / (2 * π)
    delta_hz = float(delta) / (2 * π)

    fwhm, centre = _fit_dip(
        coupling_capacitances, segment_lengths, f0 + delta_hz, gamma_hz
    )
    np.testing.assert_allclose(fwhm, gamma_hz, rtol=0.03)
    np.testing.assert_allclose(centre - f0, delta_hz, rtol=0.03, atol=20e3)


def test_decoherence_free_point() -> None:
    """Two points spaced λ/2 apart cancel the decay, Γ = 0 (Kockum Eq. 15)."""
    caps = (8e-15, 8e-15)
    f0 = _bare_frequency(caps)
    ep_eff, _ = cpw_parameters(*get_cpw_dimensions("cpw"))
    half_wavelength_um = c_0 / np.sqrt(float(jnp.real(ep_eff))) / f0 / 2 * 1e6
    gamma_full, _ = giant_atom_decay_rate_and_lamb_shift(
        f0, caps, (2 * half_wavelength_um,)
    )
    gamma_zero, delta = giant_atom_decay_rate_and_lamb_shift(
        f0, caps, (half_wavelength_um,)
    )
    assert float(gamma_zero) < 1e-9 * float(gamma_full)
    # sin(π) = 0, so the Lamb shift vanishes there as well.
    assert abs(float(delta)) < 1e-6 * float(gamma_full)

    # The SAX circuit shows no resonance dip at the decoherence-free point,
    # while λ spacing gives full extinction of S21 on resonance.
    f = jnp.linspace(f0 - 50e6, f0 + 50e6, 2001)
    for length, bound in [(half_wavelength_um, 1e-3), (2 * half_wavelength_um, None)]:
        s21 = giant_atom_transmon(
            f=f, coupling_capacitances=caps, segment_lengths=(length,)
        )
        loss = 1 - np.abs(np.asarray(s21["o1", "o2"])) ** 2
        if bound is None:
            assert loss.max() > 0.99
        else:
            assert loss.max() < bound


def test_frequency_dependent_rate_shape() -> None:
    """Γ(ω) follows 2 γ(ω) (1 + cos ω τ) for two equal points."""
    caps = (8e-15, 8e-15)
    f = jnp.linspace(4e9, 7e9, 61)
    gamma, delta = giant_atom_decay_rate_and_lamb_shift(f, caps, (30000.0,))
    gamma_1, _ = giant_atom_decay_rate_and_lamb_shift(f, caps[:1], ())
    # gamma_1 is computed with C_Σ for one coupler; rescale to two couplers.
    gamma_1 = gamma_1 * (QUBIT_CAPACITANCE + caps[0]) / (QUBIT_CAPACITANCE + sum(caps))
    ratio = np.asarray(gamma / gamma_1)
    lamb = np.asarray(delta / gamma_1)
    # Γ / γ = 2 (1 + cos ωτ) and Δ / γ = sin ωτ, so cos² + sin² = 1.
    np.testing.assert_allclose((ratio / 2 - 1) ** 2 + lamb**2, 1.0, rtol=1e-6)
    assert ratio.min() < 0.5
    assert ratio.max() > 3.5


@pytest.mark.parametrize(
    ("coupling_capacitances", "segment_lengths", "match"),
    [
        ((8e-15, 8e-15), (), "one segment fewer"),
        ((8e-15,), (100.0,), "one segment fewer"),
        (((8e-15, 8e-15),), (), "one-dimensional"),
    ],
)
def test_invalid_inputs(
    coupling_capacitances: tuple, segment_lengths: tuple, match: str
) -> None:
    """Inputs must be 1D with one segment fewer than coupling points."""
    with pytest.raises(ValueError, match=match):
        giant_atom_transmon(
            coupling_capacitances=coupling_capacitances,
            segment_lengths=segment_lengths,
        )
    with pytest.raises(ValueError, match=match):
        giant_atom_decay_rate_and_lamb_shift(
            5e9, coupling_capacitances, segment_lengths
        )


def test_default_spacing_is_kannan_decoherence_free_point() -> None:
    """The default segment is λ/2 at 4.645 GHz, Kannan et al.'s design point."""
    f_df = 4.645e9
    gamma, _ = giant_atom_decay_rate_and_lamb_shift(f_df)
    gamma_max, _ = giant_atom_decay_rate_and_lamb_shift(
        f_df, segment_lengths=(26200.0,)
    )
    assert float(gamma) < 1e-4 * float(gamma_max)
