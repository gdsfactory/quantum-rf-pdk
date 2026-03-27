"""Tests for resonator models."""

import warnings

import jax.numpy as jnp
import numpy as np

from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_frequency,
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
    s21_short = jnp.abs(r_short["coupling_o2", "coupling_o1"])
    s21_long = jnp.abs(r_long["coupling_o2", "coupling_o1"])

    f_short = f[jnp.argmin(s21_short)]
    f_long = f[jnp.argmin(s21_long)]

    assert f_long < f_short


class TestResonatorCoupled:
    """Tests for resonator_coupled with open terminations."""

    @staticmethod
    def test_with_open_start_only() -> None:
        """Test resonator_coupled with open_start=True (default), open_end=False."""
        f = jnp.array([5e9])
        result = resonator_coupled(f=f, length=2000, open_start=True, open_end=False)
        assert isinstance(result, dict)
        assert len(result) == 16  # 4 ports -> 16 S-params

    @staticmethod
    def test_with_no_open_terminations() -> None:
        """Test resonator_coupled with open_start=False and open_end=False."""
        f = jnp.array([5e9])
        result = resonator_coupled(f=f, length=2000, open_start=False, open_end=False)
        assert isinstance(result, dict)
        assert len(result) == 16

    @staticmethod
    def test_with_open_end() -> None:
        """Test resonator_coupled with open_end=True."""
        f = jnp.array([5e9])
        result = resonator_coupled(f=f, length=2000, open_start=True, open_end=True)
        assert isinstance(result, dict)
        assert len(result) == 16

    @staticmethod
    def test_with_both_open_false_structure() -> None:
        """Test that disabling both open terminations produces valid results."""
        f = jnp.linspace(3e9, 8e9, 50)

        result_open = resonator_coupled(
            f=f, length=5000, open_start=True, open_end=False
        )
        result_no_open = resonator_coupled(
            f=f, length=5000, open_start=False, open_end=False
        )

        # Both configurations should produce valid finite S-parameters
        for key in result_open:
            assert jnp.all(jnp.isfinite(result_open[key]))
        for key in result_no_open:
            assert jnp.all(jnp.isfinite(result_no_open[key]))


class TestResonatorFrequency:
    """Tests for resonator_frequency function."""

    @staticmethod
    def test_quarter_wave_frequency() -> None:
        """Test quarter-wave resonator frequency calculation."""
        # For a CPW with ep_eff ≈ 6.35, c₀ = 3e8 m/s
        # f_qw = v_p / (4L) where v_p = c₀/sqrt(ep_eff)
        length_um = 5000.0
        f_qw = resonator_frequency(length=length_um, is_quarter_wave=True)

        assert f_qw > 0, "Frequency should be positive"
        # For typical Si CPW (ep_eff ~ 6-7), expect f ~ 3-4 GHz for 5mm
        assert 2e9 < f_qw < 10e9, (
            f"Quarter-wave frequency {f_qw / 1e9:.2f} GHz out of range"
        )

    @staticmethod
    def test_half_wave_frequency() -> None:
        """Test half-wave resonator frequency calculation."""
        length_um = 5000.0
        f_qw = resonator_frequency(length=length_um, is_quarter_wave=True)
        f_hw = resonator_frequency(length=length_um, is_quarter_wave=False)

        # Half-wave should be twice the quarter-wave frequency
        np.testing.assert_allclose(f_hw, 2 * f_qw, rtol=1e-6)

    @staticmethod
    def test_with_explicit_epsilon_eff() -> None:
        """Test with explicitly provided epsilon_eff."""
        length_um = 5000.0
        ep_eff = 6.0
        f = resonator_frequency(length=length_um, epsilon_eff=ep_eff)
        # v_p = c₀/sqrt(6.0), f = v_p / (4 * 5e-3)
        expected = 3e8 / np.sqrt(6.0) / (4 * 5e-3)
        # Allow for numerical tolerance due to constant precision differences
        np.testing.assert_allclose(f, expected, rtol=1e-2)

    @staticmethod
    def test_with_deprecated_media() -> None:
        """Test with deprecated media parameter."""
        from unittest.mock import Mock

        mock_media = Mock()
        mock_media.ep_r = jnp.array([11.7])

        length_um = 5000.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f = resonator_frequency(length=length_um, media=mock_media)
            assert f > 0
            # Should issue a deprecation warning
            assert any(issubclass(wi.category, DeprecationWarning) for wi in w)

    @staticmethod
    def test_frequency_inversely_proportional_to_length() -> None:
        """Test that frequency decreases as length increases."""
        f_short = resonator_frequency(length=3000.0)
        f_long = resonator_frequency(length=6000.0)
        assert f_short > f_long, "Shorter resonator should have higher frequency"
