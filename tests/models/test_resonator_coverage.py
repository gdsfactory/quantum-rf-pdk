"""Additional tests for qpdk.models.resonator module - covering missing lines."""

import warnings

import jax.numpy as jnp
import numpy as np

from qpdk.models.resonator import resonator_coupled, resonator_frequency


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
        """Test resonator_coupled with open_start=False and open_end=False (line 139)."""
        f = jnp.array([5e9])
        result = resonator_coupled(f=f, length=2000, open_start=False, open_end=False)
        assert isinstance(result, dict)
        assert len(result) == 16

    @staticmethod
    def test_with_open_end() -> None:
        """Test resonator_coupled with open_end=True (lines 142-144)."""
        f = jnp.array([5e9])
        result = resonator_coupled(f=f, length=2000, open_start=True, open_end=True)
        assert isinstance(result, dict)
        assert len(result) == 16

    @staticmethod
    def test_with_both_open_false_structure() -> None:
        """Test that disabling both open terminations produces valid results."""
        f = jnp.linspace(3e9, 8e9, 50)

        result_open = resonator_coupled(f=f, length=5000, open_start=True, open_end=False)
        result_no_open = resonator_coupled(f=f, length=5000, open_start=False, open_end=False)

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
        assert 2e9 < f_qw < 10e9, f"Quarter-wave frequency {f_qw / 1e9:.2f} GHz out of range"

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
        """Test with deprecated media parameter (lines 184-188)."""
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
