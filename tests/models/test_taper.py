"""Tests for qpdk.models.waveguides.taper_cross_section function."""

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import assume, given, settings

from qpdk.models.waveguides import launcher, straight, taper_cross_section
from qpdk.tech import coplanar_waveguide

MAX_EXAMPLES = 50


class TestTaperWaveguide:
    """Unit and integration tests for tapered waveguide model."""

    def test_taper_default_parameters(self) -> None:
        """Test taper with default parameters."""
        result = taper_cross_section()
        assert isinstance(result, dict)
        assert ("o1", "o1") in result
        assert ("o1", "o2") in result
        assert ("o2", "o1") in result
        assert ("o2", "o2") in result

    def test_taper_returns_stype(self) -> None:
        """Test that taper returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = taper_cross_section(f=f, length=1000)

        assert isinstance(result, dict)
        expected_keys = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}
        assert set(result.keys()) == expected_keys
        for value in result.values():
            assert hasattr(value, "__len__")

    def test_taper_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 10
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = taper_cross_section(f=f, length=200)

        for value in result.values():
            assert len(value) == n_freq

    def test_taper_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 7e9, 50)
        cs1 = coplanar_waveguide(width=10, gap=6)
        cs2 = coplanar_waveguide(width=20, gap=10)
        result = taper_cross_section(
            f=f, length=150, cross_section_1=cs1, cross_section_2=cs2
        )

        s12 = result[("o1", "o2")]
        s21 = result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-9

    def test_taper_single_frequency(self) -> None:
        """Test taper with a single frequency point."""
        f = jnp.array([5e9])
        result = taper_cross_section(f=f, length=100)

        assert isinstance(result, dict)
        for value in result.values():
            assert len(value) == 1

    @given(
        n_freq=st.integers(min_value=1, max_value=10),
        f_min=st.floats(min_value=0.5e9, max_value=5e9),
        f_max=st.floats(min_value=5.1e9, max_value=10e9),
        length=st.floats(min_value=1, max_value=1000),
        width1=st.floats(min_value=5, max_value=15),
        gap1=st.floats(min_value=3, max_value=8),
        width2=st.floats(min_value=5, max_value=15),
        gap2=st.floats(min_value=3, max_value=8),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_taper_with_hypothesis(
        self,
        n_freq: int,
        f_min: float,
        f_max: float,
        length: float,
        width1: float,
        gap1: float,
        width2: float,
        gap2: float,
    ) -> None:
        """Test taper with random valid parameters using hypothesis."""
        assume(f_max > f_min)

        f = jnp.linspace(f_min, f_max, n_freq)
        cs1 = coplanar_waveguide(width=width1, gap=gap1)
        cs2 = coplanar_waveguide(width=width2, gap=gap2)
        result = taper_cross_section(
            f=f, length=length, cross_section_1=cs1, cross_section_2=cs2
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        for value in result.values():
            assert len(value) == n_freq

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        length=st.floats(min_value=10, max_value=5000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_taper_passivity(self, f_center: float, length: float) -> None:
        """Test that the taper satisfies passivity (energy conservation)."""
        f = jnp.linspace(f_center * 0.9, f_center * 1.1, 10)
        cs1 = coplanar_waveguide(width=10, gap=6)
        cs2 = coplanar_waveguide(width=20, gap=10)
        result = taper_cross_section(
            f=f, length=length, cross_section_1=cs1, cross_section_2=cs2
        )

        s11 = result[("o1", "o1")]
        s21 = result[("o2", "o1")]

        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission

        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_taper_identical_cross_sections(self) -> None:
        """Test that a taper with identical cross-sections behaves like a straight waveguide."""
        f = jnp.linspace(4e9, 6e9, 20)
        length = 500
        cs = coplanar_waveguide(width=10, gap=6)

        taper_result = taper_cross_section(
            f=f, length=length, cross_section_1=cs, cross_section_2=cs
        )
        straight_result = straight(f=f, length=length, cross_section=cs)

        s21_taper = taper_result[("o2", "o1")]
        s21_straight = straight_result[("o2", "o1")]

        max_diff = jnp.max(jnp.abs(s21_taper - s21_straight))
        assert max_diff < 1e-6, (
            "Taper with same start/end CS should match straight waveguide"
        )

    def test_taper_zero_length(self) -> None:
        """Test taper with zero length (should be a through connection)."""
        f = jnp.array([5e9])
        cs1 = coplanar_waveguide(width=10, gap=6)
        cs2 = coplanar_waveguide(width=20, gap=10)
        # Even with different cross-sections, zero length should be a perfect through
        # because the model doesn't account for abrupt junction effects.
        result = taper_cross_section(
            f=f, length=0, cross_section_1=cs1, cross_section_2=cs2
        )

        s21 = result[("o2", "o1")]
        transmission = jnp.abs(s21)[0]
        s11 = result[("o1", "o1")]
        reflection = jnp.abs(s11)[0]

        # Should be very close to 1 (perfect transmission)
        assert transmission > 0.999, (
            f"Zero length should have ~perfect transmission, got {transmission}"
        )
        # Should be very close to 0 (no reflection)
        assert reflection < 0.001, (
            f"Zero length should have ~zero reflection, got {reflection}"
        )


class TestLauncher:
    """Unit and integration tests for launcher model."""

    def test_launcher_default_parameters(self) -> None:
        """Test launcher with default parameters."""
        result = launcher()
        assert isinstance(result, dict)
        assert ("waveport", "waveport") in result
        assert ("waveport", "o1") in result
        assert ("o1", "waveport") in result
        assert ("o1", "o1") in result

    def test_launcher_returns_stype(self) -> None:
        """Test that launcher returns a valid sax.SType dictionary."""
        f = jnp.array([5e9, 6e9, 7e9])
        result = launcher(f=f)
        assert isinstance(result, dict)
        expected_keys = {
            ("waveport", "waveport"),
            ("waveport", "o1"),
            ("o1", "waveport"),
            ("o1", "o1"),
        }
        assert set(result.keys()) == expected_keys

    def test_launcher_output_shape(self) -> None:
        """Test that output array shapes match input frequency array length."""
        n_freq = 15
        f = jnp.linspace(4e9, 8e9, n_freq)
        result = launcher(f=f)
        for value in result.values():
            assert len(value) == n_freq

    def test_launcher_reciprocity(self) -> None:
        """Test that S-parameters are reciprocal (S12 = S21)."""
        f = jnp.linspace(3e9, 7e9, 50)
        result = launcher(f=f)
        s12 = result[("waveport", "o1")]
        s21 = result[("o1", "waveport")]
        max_diff = jnp.max(jnp.abs(s12 - s21))
        assert max_diff < 1e-9

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        straight_length=st.floats(min_value=10, max_value=1000),
        taper_length=st.floats(min_value=10, max_value=1000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_launcher_passivity(
        self, f_center: float, straight_length: float, taper_length: float
    ) -> None:
        """Test that the launcher satisfies passivity."""
        f = jnp.linspace(f_center * 0.9, f_center * 1.1, 10)
        result = launcher(
            f=f, straight_length=straight_length, taper_length=taper_length
        )
        s11 = result[("waveport", "waveport")]
        s21 = result[("o1", "waveport")]
        power_reflection = jnp.abs(s11) ** 2
        power_transmission = jnp.abs(s21) ** 2
        total_power = power_reflection + power_transmission
        assert jnp.all(total_power <= 1.0 + 1e-6)

    def test_launcher_zero_length(self) -> None:
        """Test launcher with zero length (should be a through connection)."""
        f = jnp.array([5e9])
        cs_big = coplanar_waveguide(width=200, gap=100)
        cs_small = coplanar_waveguide(width=10, gap=6)
        result = launcher(
            f=f,
            straight_length=0,
            taper_length=0,
            cross_section_big=cs_big,
            cross_section_small=cs_small,
        )

        s21 = result[("o1", "waveport")]
        transmission = jnp.abs(s21)[0]
        s11 = result[("waveport", "waveport")]
        reflection = jnp.abs(s11)[0]

        assert transmission > 0.999
        assert reflection < 0.001

    @given(
        n_freq=st.integers(min_value=1, max_value=10),
        f_min=st.floats(min_value=0.5e9, max_value=5e9),
        f_max=st.floats(min_value=5.1e9, max_value=10e9),
        straight_length=st.floats(min_value=1, max_value=1000),
        taper_length=st.floats(min_value=1, max_value=1000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_launcher_with_hypothesis(
        self,
        n_freq: int,
        f_min: float,
        f_max: float,
        straight_length: float,
        taper_length: float,
    ) -> None:
        """Test launcher with random valid parameters using hypothesis."""
        assume(f_max > f_min)

        f = jnp.linspace(f_min, f_max, n_freq)
        cs_big = coplanar_waveguide(width=200, gap=100)
        cs_small = coplanar_waveguide(width=10, gap=6)
        result = launcher(
            f=f,
            straight_length=straight_length,
            taper_length=taper_length,
            cross_section_big=cs_big,
            cross_section_small=cs_small,
        )

        assert isinstance(result, dict)
        assert len(result) == 4

        for value in result.values():
            assert len(value) == n_freq
