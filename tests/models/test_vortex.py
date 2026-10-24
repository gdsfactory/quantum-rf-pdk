"""Tests for :mod:`qpdk.models.vortex`."""

from typing import final

import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.models.vortex import (
    _cascade,
    cpw_bulge,
    neck_termination,
    resonator_half_wave_bulge,
    resonator_quarter_wave_neck,
)
from qpdk.models.waveguides import straight, straight_shorted

from .base import OnePortModelTestSuite, TwoPortModelTestSuite


@final
class TestNeckTermination(OnePortModelTestSuite):
    """Tests for the neck termination model."""

    model_function = neck_termination


@final
class TestCpwBulge(TwoPortModelTestSuite):
    """Tests for the bulge model."""

    model_function = cpw_bulge


@final
class TestResonatorQuarterWaveNeck(OnePortModelTestSuite):
    """Tests for the quarter-wave neck resonator model."""

    model_function = resonator_quarter_wave_neck


@final
class TestResonatorHalfWaveBulge(TwoPortModelTestSuite):
    """Tests for the half-wave bulge resonator model."""

    model_function = resonator_half_wave_bulge


_F = jnp.linspace(3e9, 9e9, 20001)


def _shorted_resonance(s11: jnp.ndarray) -> float:
    """First frequency where the input reactance of a shorted line changes sign."""
    s = np.asarray(s11)
    z = (1 + s) / (1 - s)
    idx = np.flatnonzero(np.diff(np.sign(np.imag(z))) != 0)
    assert idx.size, "no resonance found in the sweep"
    return float(_F[idx[0]])


class TestPhysics:
    """Physical sanity checks for the vortex models."""

    @staticmethod
    def test_uniform_cascade_matches_straight() -> None:
        """A section of the port cross-section reproduces ``straight``."""
        f = jnp.array([4e9, 6e9])
        ref = straight(f=f, length=1000)
        got = _cascade(f, "cpw", [(1000, 10, 6)])
        np.testing.assert_allclose(got["o2", "o1"], ref["o2", "o1"], atol=1e-12)
        np.testing.assert_allclose(got["o1", "o1"], ref["o1", "o1"], atol=1e-12)

    @staticmethod
    def test_neck_lowers_resonance_more_than_its_length() -> None:
        """The high-impedance neck lowers f0 below a plain line of the same length."""
        f_plain = _shorted_resonance(straight_shorted(f=_F, length=4000)["o1", "o1"])
        f_same_length = _shorted_resonance(
            straight_shorted(f=_F, length=4000 + 100 + 325)["o1", "o1"]
        )
        f_neck = _shorted_resonance(resonator_quarter_wave_neck(f=_F)["o1", "o1"])
        assert f_neck < f_same_length < f_plain

    @staticmethod
    def test_narrower_neck_lowers_resonance() -> None:
        """A narrower neck has higher inductance and a lower resonance."""
        f_narrow = _shorted_resonance(
            resonator_quarter_wave_neck(f=_F, neck_width=1)["o1", "o1"]
        )
        f_wide = _shorted_resonance(
            resonator_quarter_wave_neck(f=_F, neck_width=8)["o1", "o1"]
        )
        assert f_narrow < f_wide

    @staticmethod
    def test_taper_shape_changes_response() -> None:
        """The smooth and linear tapers give different responses."""
        f = jnp.array([5e9])
        linear = neck_termination(f=f)["o1", "o1"]
        smooth = neck_termination(f=f, taper_shape="smooth")["o1", "o1"]
        assert not jnp.allclose(linear, smooth)

    @staticmethod
    def test_bulge_reflects() -> None:
        """The wider, lower-impedance bulge gives a small but finite reflection."""
        s = cpw_bulge(f=jnp.array([5e9, 6e9]))
        s11 = np.abs(np.asarray(s["o1", "o1"]))
        assert np.all(s11 > 1e-3)
        assert np.all(s11 < 0.1)

    @staticmethod
    def test_half_wave_too_short_raises() -> None:
        """A half-wave resonator shorter than the bulge is rejected."""
        with pytest.raises(ValueError, match="must exceed the bulge length"):
            resonator_half_wave_bulge(f=jnp.array([5e9]), length=50)

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"neck_width": 10}, "neck_width < CPW width"),
            ({"neck_width": 0}, "neck_width < CPW width"),
            ({"window_width": 15}, "CPW footprint"),
        ],
    )
    def test_neck_outside_window_raises(kwargs: dict, match: str) -> None:
        """Geometries the layout cell refuses are refused by the model too."""
        with pytest.raises(ValueError, match=match):
            neck_termination(f=jnp.array([5e9]), **kwargs)

    @staticmethod
    def test_bulge_raises_half_wave_resonance() -> None:
        """A low-inductance bulge at the current antinode raises f0."""
        f = np.linspace(6e9, 9e9, 30001)

        def open_open_resonance(s: dict) -> float:
            # Leave o2 open and find where the input admittance at o1 vanishes.
            s11, s21, s12, s22 = (
                np.asarray(s[k])
                for k in [("o1", "o1"), ("o2", "o1"), ("o1", "o2"), ("o2", "o2")]
            )
            gamma = s11 + s12 * s21 / (1 - s22)
            y = (1 - gamma) / (1 + gamma)
            b = np.imag(y)
            # Zeros, not poles: the sign flips while |B| stays small.
            idx = np.flatnonzero(
                (np.diff(np.sign(b)) != 0) & (np.abs(b[:-1]) < 1) & (np.abs(b[1:]) < 1)
            )
            assert idx.size, "no resonance found in the sweep"
            return float(f[idx[0]])

        f_plain = open_open_resonance(straight(f=jnp.asarray(f), length=8000))
        f_bulge = open_open_resonance(resonator_half_wave_bulge(f=jnp.asarray(f)))
        assert f_plain < f_bulge < 1.01 * f_plain
