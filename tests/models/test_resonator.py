"""Tests for resonator models."""

import jax.numpy as jnp

from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_half_wave,
    resonator_quarter_wave,
)


def test_resonator_models_port_count() -> None:
    f = jnp.array([5e9])

    # Half wave and basic resonator are 2-port
    r = resonator(f=f, length=2000)
    assert len(r) == 4  # 2 ports -> 4 S-parameters

    r_half = resonator_half_wave(f=f, length=2000)
    assert len(r_half) == 4

    # Quarter wave is 1-port, but our model exposes a dummy 2nd port for layout consistency.
    # Actually evaluate_circuit drops the disconnected dummy port, so it only has 1 S-parameter
    r_quarter = resonator_quarter_wave(f=f, length=2000)
    assert len(r_quarter) == 1  # 1 port -> 1 S-parameter


def test_resonator_coupled_basic_structure() -> None:
    f = jnp.array([5e9])
    rc = resonator_coupled(f=f, length=2000)
    assert len(rc) == 16  # 4 ports -> 16 S-parameters


def test_resonator_frequency_shifts_with_length() -> None:
    # Use quarter_wave_resonator_coupled to find resonance (notch filter on the probeline)
    # Quarter-wave resonance for 8000 um is ~3.75 GHz, for 10000 um is ~3.0 GHz
    f = jnp.linspace(2e9, 5e9, 1000)

    r_short = quarter_wave_resonator_coupled(f=f, length=8000)
    r_long = quarter_wave_resonator_coupled(f=f, length=10000)

    # Find min transmission (S21 of probeline) to get resonance
    s21_short = jnp.abs(r_short[("coupling_o2", "coupling_o1")])
    s21_long = jnp.abs(r_long[("coupling_o2", "coupling_o1")])

    f_short = f[jnp.argmin(s21_short)]
    f_long = f[jnp.argmin(s21_long)]

    assert f_long < f_short
