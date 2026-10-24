"""Tests for the vacuum-gap drum capacitor and drum LC resonator models."""

from typing import final, override

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

from qpdk.models.constants import ε_0, π
from qpdk.models.drum import (
    DRUM_MECHANICAL_FREQUENCY,
    VACUUM_GAP,
    drum_effective_mass,
    drum_electromechanical_coupling,
    drum_lc_resonator,
    drum_single_photon_coupling,
    drum_zero_point_motion,
    vacuum_gap_capacitance_analytical,
    vacuum_gap_capacitor,
)
from qpdk.models.inductor import meander_inductor_inductance_analytical
from qpdk.tech import get_vacuum_gap

from .base import TwoPortModelTestSuite

MAX_EXAMPLES = 20


@final
class TestVacuumGapCapacitor(TwoPortModelTestSuite):
    """Standard S-parameter checks for vacuum_gap_capacitor."""

    model_function = staticmethod(vacuum_gap_capacitor)

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        """Get model-specific keyword arguments."""
        return {"radius": 10.0, "gap": 0.1}


@final
class TestDrumLCResonator(TwoPortModelTestSuite):
    """Standard S-parameter checks for drum_lc_resonator."""

    model_function = staticmethod(drum_lc_resonator)
    freq_range = (1e9, 20e9)


def test_default_gap_comes_from_layer_stack() -> None:
    """The default gap is the 50 nm VacuumGap level of the layer stack."""
    assert get_vacuum_gap() == pytest.approx(0.05)
    assert pytest.approx(0.05) == VACUUM_GAP


def test_parallel_plate_capacitance() -> None:
    """Without fringing, C = ε0 π a² / d for the 15 µm drum at 50 nm."""
    capacitance = vacuum_gap_capacitance_analytical(radius=7.5, gap=0.05, fringing=0)
    expected = ε_0 * π * (7.5e-6) ** 2 / 50e-9
    assert float(capacitance) == pytest.approx(expected, rel=1e-12)
    assert float(capacitance) == pytest.approx(31.29e-15, rel=1e-3)


@given(
    radius=st.floats(min_value=1.0, max_value=100.0),
    gap=st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_fringing_is_small_positive_correction(radius: float, gap: float) -> None:
    """Fringing adds capacitance, by less than the bare value for d ≪ a."""
    bare = float(vacuum_gap_capacitance_analytical(radius, gap, fringing=0))
    total = float(vacuum_gap_capacitance_analytical(radius, gap))
    assert bare < total < 2 * bare


def test_teufel_lc_design_values() -> None:
    """C ≈ 38 fF and L ≈ 12 nH resonate near the quoted 7.5 GHz."""
    frequency = 1 / (2 * π * jnp.sqrt(12e-9 * 38e-15))
    assert float(frequency) == pytest.approx(7.5e9, rel=0.01)


def test_teufel_coupling_design_value() -> None:
    """G/2π ≈ 100 MHz/nm for a 10 GHz cavity and a 50 nm gap."""
    coupling = drum_electromechanical_coupling(resonance_frequency=10e9, gap=0.05)
    assert float(coupling) / (2 * π) * 1e-9 == pytest.approx(-100e6, rel=1e-12)


def test_coupling_scales_with_participation() -> None:
    """G = -η ωc / (2d) scales linearly with the participation ratio."""
    full = drum_electromechanical_coupling(7.5e9, gap=0.05)
    partial = drum_electromechanical_coupling(7.5e9, gap=0.05, participation=0.25)
    assert float(partial) == pytest.approx(0.25 * float(full))


def test_teufel_zero_point_motion() -> None:
    """The 15 µm, 100 nm Al disc gives x_zp ≈ 4.1 fm at 10.69 MHz."""
    mass = drum_effective_mass(radius=7.5)
    assert float(mass) == pytest.approx(47.7e-15, rel=1e-3)
    x_zp = drum_zero_point_motion(mass, DRUM_MECHANICAL_FREQUENCY)
    assert float(x_zp) == pytest.approx(4.1e-15, rel=0.02)


def test_single_photon_coupling_is_product() -> None:
    """g0 = |G| x_zp."""
    mass = drum_effective_mass(radius=7.5)
    g0 = drum_single_photon_coupling(resonance_frequency=7.5e9, effective_mass=mass)
    expected = abs(float(drum_electromechanical_coupling(7.5e9))) * float(
        drum_zero_point_motion(mass)
    )
    assert float(g0) == pytest.approx(expected)


def test_resonator_dip_at_lc_frequency() -> None:
    """The hanger transmission vanishes at 1/(2π √(L (C_d + C_c)))."""
    inductance = meander_inductor_inductance_analytical(
        n_turns=71,
        turn_length=300.0,
        wire_width=2.0,
        wire_gap=4.0,
        sheet_inductance=0.4e-12,
    )
    capacitance = vacuum_gap_capacitance_analytical(radius=7.5, gap=VACUUM_GAP)
    f_r = 1 / (2 * π * jnp.sqrt(inductance * (capacitance + 1e-15)))
    f = jnp.array([0.9, 1.0, 1.1]) * f_r
    s21 = jnp.abs(drum_lc_resonator(f=f)["o2", "o1"])
    assert float(s21[1]) < 1e-2
    assert float(s21[0]) > 0.99
    assert float(s21[2]) > 0.99


def test_smaller_gap_lowers_resonance() -> None:
    """A smaller gap raises the drum capacitance and lowers the resonance."""
    c_wide = vacuum_gap_capacitance_analytical(radius=7.5, gap=0.06)
    c_narrow = vacuum_gap_capacitance_analytical(radius=7.5, gap=0.04)
    assert float(c_narrow) > float(c_wide)


def test_helpers_jit_and_differentiate() -> None:
    """The helpers jit and dC/dd matches -ε0 π a² / d² without fringing."""
    capacitance = jax.jit(vacuum_gap_capacitance_analytical)
    assert jnp.isfinite(capacitance(7.5, 0.05))
    dc_dd = jax.grad(lambda d: vacuum_gap_capacitance_analytical(7.5, d, 0.0))(0.05)
    expected = -ε_0 * π * (7.5e-6) ** 2 / (50e-9) ** 2 * 1e-6
    assert float(dc_dd) == pytest.approx(expected, rel=1e-6)
