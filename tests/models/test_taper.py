"""Tests for qpdk.models.waveguides.taper_cross_section function."""

from typing import ClassVar, final

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import given, settings

from qpdk.models.waveguides import launcher, straight, taper_cross_section
from qpdk.tech import coplanar_waveguide

from .base import BaseModelTestSuite, TwoPortModelTestSuite

MAX_EXAMPLES = 50


@final
class TestTaperWaveguide(TwoPortModelTestSuite):
    """Unit and integration tests for tapered waveguide model."""

    model_function = taper_cross_section

    def get_model_kwargs(self) -> dict:
        """Get model-specific keyword arguments."""
        return {"length": 200}

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        length=st.floats(min_value=10, max_value=5000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_passivity_hypothesis(self, f_center: float, length: float) -> None:
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

    def test_identical_cross_sections(self) -> None:
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

    def test_zero_length(self) -> None:
        """Test taper with zero length (should be a through connection)."""
        f = jnp.array([5e9])
        cs1 = coplanar_waveguide(width=10, gap=6)
        cs2 = coplanar_waveguide(width=20, gap=10)
        result = taper_cross_section(
            f=f, length=0, cross_section_1=cs1, cross_section_2=cs2
        )

        s21 = result[("o2", "o1")]
        transmission = jnp.abs(s21)[0]
        s11 = result[("o1", "o1")]
        reflection = jnp.abs(s11)[0]

        assert transmission > 0.999, (
            f"Zero length should have ~perfect transmission, got {transmission}"
        )
        assert reflection < 0.001, (
            f"Zero length should have ~zero reflection, got {reflection}"
        )


@final
class TestLauncher(BaseModelTestSuite):
    """Unit and integration tests for launcher model."""

    model_function = launcher
    expected_ports: ClassVar[set[str]] = {"waveport", "o1"}

    def get_frequency_array(self, n_points: int | None = None) -> jnp.ndarray:
        """Generate a frequency array for testing."""
        n = n_points if n_points is not None else self.n_freq_default
        return jnp.linspace(*self.freq_range, n)

    def get_model_kwargs(self) -> dict:
        """Get model-specific keyword arguments."""
        return {"straight_length": 100, "taper_length": 100}

    @given(
        f_center=st.floats(min_value=1e9, max_value=9e9),
        straight_length=st.floats(min_value=10, max_value=1000),
        taper_length=st.floats(min_value=10, max_value=1000),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_passivity_hypothesis(
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

    def test_zero_length(self) -> None:
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
