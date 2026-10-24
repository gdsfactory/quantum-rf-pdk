"""Tests for the Josephson junction array model."""

import itertools
from functools import partial
from typing import final, override

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
import sax
from hypothesis import given, settings
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from qpdk.models.constants import Φ_0
from qpdk.models.generic import series_impedance, shunt_admittance
from qpdk.models.junction import josephson_junction_array

from .base import TwoPortModelTestSuite

L_J = 1.9e-9
C_J = 40e-15
IC = Φ_0 / (2 * jnp.pi * L_J)


def _rcsj_impedance(
    f: jax.Array, ic: float, capacitance: float, resistance: float, ib: float = 0.0
) -> jax.Array:
    """Impedance of the linearised RCSJ element of ``josephson_junction``."""
    ω = 2 * jnp.pi * f
    l_j = Φ_0 / (2 * jnp.pi * ic * jnp.sqrt(1 - (ib / ic) ** 2))
    return 1 / (1 / resistance + 1j * ω * capacitance + 1 / (1j * ω * l_j))


def _abcd_b(s: sax.SDict, z0: float = 50.0) -> jax.Array:
    """ABCD ``B`` element of a two-port from its S-parameters (Pozar Table 4.2)."""
    s11, s12, s21, s22 = s["o1", "o1"], s["o1", "o2"], s["o2", "o1"], s["o2", "o2"]
    return z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)


def _analytic_modes(n: int, l_j: float, c_j: float, c_g: float) -> list[float]:
    """Mode frequencies in Hz of the grounded ideal chain, Masluk et al. Eq. (7)."""
    ω_p = 1 / jnp.sqrt(l_j * c_j)
    k = jnp.arange(1, n)
    x = 1 - jnp.cos(jnp.pi * k / n)
    return [
        float(f_k) for f_k in ω_p * jnp.sqrt(x / (c_g / (2 * c_j) + x)) / (2 * jnp.pi)
    ]


@final
class TestJosephsonJunctionArray(TwoPortModelTestSuite):
    """Passivity, reciprocity and shape checks for ``josephson_junction_array``."""

    model_function = josephson_junction_array

    @override
    @staticmethod
    def get_model_kwargs() -> dict:
        return {"n_junctions": 20, "resistance": 1e6}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ic": 1e-6, "capacitance": 5e-15, "resistance": 10e3},
        {"ic": 2e-7, "capacitance": 40e-15, "resistance": 1e5, "ib": 1e-7},
    ],
)
def test_single_junction_limit(kwargs: dict) -> None:
    """N = 1 is one RCSJ junction in series, referenced to 50 Ω."""
    f = jnp.linspace(1e9, 20e9, 41)
    s_array = josephson_junction_array(f=f, n_junctions=1, **kwargs)
    s_single = series_impedance(f=f, z=_rcsj_impedance(f, **kwargs))
    for key, value in s_single.items():
        assert_allclose(s_array[key], value, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n_junctions", [2, 3, 5])
def test_matches_sax_netlist(n_junctions: int) -> None:
    """The ABCD cascade matches a SAX circuit of junctions and island capacitors."""
    f = jnp.linspace(1e9, 25e9, 31)
    c_g = 2e-15
    resistance = 1e5
    z_j = _rcsj_impedance(f, IC, C_J, resistance)

    instances = {f"j{i}": "junction" for i in range(n_junctions)}
    instances |= {f"g{i}": "island" for i in range(n_junctions - 1)}
    connections = {}
    for i in range(n_junctions - 1):
        connections[f"j{i},o2"] = f"g{i},o1"
        connections[f"g{i},o2"] = f"j{i + 1},o1"
    circuit, _ = sax.circuit(
        netlist={
            "instances": instances,
            "connections": connections,
            "ports": {"o1": "j0,o1", "o2": f"j{n_junctions - 1},o2"},
        },
        models={
            "junction": partial(series_impedance, z=z_j),
            "island": partial(shunt_admittance, y=2j * jnp.pi * f * c_g),
        },
    )
    s_netlist = circuit(f=f)
    s_array = josephson_junction_array(
        f=f,
        n_junctions=n_junctions,
        ic=IC,
        capacitance=C_J,
        ground_capacitance=c_g,
        resistance=resistance,
    )
    for key in [("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")]:
        assert_allclose(s_array[key], s_netlist[key], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("n_junctions", "c_g"), [(10, 1e-15), (10, 0.04e-15), (33, 2e-15)]
)
def test_dispersion_matches_analytic(n_junctions: int, c_g: float) -> None:
    """Shorted-chain resonances (zeros of ABCD ``B``) follow Masluk et al. Eq. (7)."""

    def im_b(f: float) -> float:
        s = josephson_junction_array(
            f=jnp.array([f]),
            n_junctions=n_junctions,
            ic=IC,
            capacitance=C_J,
            ground_capacitance=c_g,
        )
        return float(jnp.imag(_abcd_b(s))[0])

    modes = _analytic_modes(n_junctions, L_J, C_J, c_g)
    # Bracket each root by a tenth of the gap to its neighbours; for small C_g
    # the modes crowd just below the plasma frequency.
    gaps = [b - a for a, b in itertools.pairwise(modes)]
    for k, f_k in enumerate(modes):
        half_width = 0.1 * min(gaps[max(k - 1, 0)], gaps[min(k, len(gaps) - 1)])
        root = brentq(im_b, f_k - half_width, f_k + half_width, xtol=1e-3)
        assert root == pytest.approx(f_k, rel=1e-8)


def test_plasma_frequency_cutoff() -> None:
    """Transmission vanishes at the single-junction plasma frequency 1/√(L_J C_J)."""
    f_p = 1 / (2 * jnp.pi * jnp.sqrt(L_J * C_J))
    assert float(f_p) == pytest.approx(18.26e9, rel=1e-3)
    s = josephson_junction_array(f=jnp.array([f_p]), n_junctions=5)
    assert float(jnp.abs(s["o2", "o1"])[0]) < 1e-6


def test_masluk_lowest_self_resonance() -> None:
    """The fit parameters give the 14.2 GHz lowest mode Masluk et al. report for N = 80."""
    s_grid = jnp.linspace(13e9, 15e9, 2001)
    s = josephson_junction_array(f=s_grid, n_junctions=80)
    im_b = jnp.imag(_abcd_b(s))
    crossing = jnp.flatnonzero(jnp.diff(jnp.sign(im_b)) != 0)
    assert crossing.size == 1
    assert float(s_grid[crossing[0]]) == pytest.approx(14.2e9, rel=5e-3)


def test_low_frequency_inductance() -> None:
    """Well below the self-resonances the chain is a series inductance N L_J."""
    f = jnp.array([10e6])
    n = 43
    s = josephson_junction_array(f=f, n_junctions=n)
    z = _abcd_b(s)
    l_eff = jnp.imag(z) / (2 * jnp.pi * f)
    assert float(l_eff[0]) == pytest.approx(n * L_J, rel=1e-4)


def test_large_array_is_jittable_and_finite() -> None:
    """A 1000-junction chain compiles under jit and stays finite."""
    f = jnp.linspace(1e9, 10e9, 11)
    result = jax.jit(lambda f: josephson_junction_array(f=f, n_junctions=1000))(f)
    assert all(jnp.all(jnp.isfinite(value)) for value in result.values())


def test_invalid_n_junctions() -> None:
    with pytest.raises(ValueError, match="n_junctions"):
        josephson_junction_array(n_junctions=0)


@given(
    n_junctions=st.integers(min_value=1, max_value=60),
    c_g=st.floats(min_value=1e-18, max_value=1e-14),
    resistance=st.floats(min_value=1e3, max_value=1e9),
)
@settings(deadline=None, max_examples=20)
def test_passive_and_reciprocal_hypothesis(
    n_junctions: int, c_g: float, resistance: float
) -> None:
    """The chain is reciprocal and passive for any lossy or lossless parameters."""
    f = jnp.linspace(1e9, 30e9, 15)
    s = josephson_junction_array(
        f=f, n_junctions=n_junctions, ground_capacitance=c_g, resistance=resistance
    )
    assert_allclose(s["o1", "o2"], s["o2", "o1"])
    for port in ("o1", "o2"):
        power = jnp.abs(s[port, "o1"]) ** 2 + jnp.abs(s[port, "o2"]) ** 2
        assert jnp.all(power <= 1 + 1e-9)
