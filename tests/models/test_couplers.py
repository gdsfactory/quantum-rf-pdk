"""Tests for qpdk.models.couplers module."""

import inspect
from typing import final

import gdsfactory as gf
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from qpdk.cells.waveguides import coupler_ring as coupler_ring_cell
from qpdk.models.constants import π
from qpdk.models.couplers import (
    _access_line_s_params,
    coupler_ring,
    coupler_straight,
    cpw_cpw_coupling_capacitance,
    cpw_cpw_coupling_capacitance_per_length_analytical,
)
from qpdk.models.cpw import cpw_z0_from_cross_section
from qpdk.models.waveguides import straight
from qpdk.tech import coplanar_waveguide

from .base import FourPortModelTestSuite


@final
class TestCouplerRing(FourPortModelTestSuite):
    """Tests for coupler_ring model."""

    model_function = coupler_ring


@final
class TestCouplerRingCustomGeometry(FourPortModelTestSuite):
    """Tests for coupler_ring model with explicit non-default access geometry."""

    model_function = coupler_ring

    @staticmethod
    def get_model_kwargs() -> dict:
        """Exercise the cascaded access lines in the shared port checks."""
        return {"radius": 50.0, "length_extension": 7.0}


@final
class TestCouplerStraight(FourPortModelTestSuite):
    """Tests for coupler_straight model."""

    model_function = coupler_straight


def test_coupler_ring_default_geometry_matches_layout_cell() -> None:
    """Resolve default access geometry the same way the layout cell does."""
    radius = gf.get_cross_section("cpw").radius
    assert radius is not None
    length_extension = 3.0 + radius
    length_x = inspect.signature(coupler_ring).parameters["length_x"].default

    layout = coupler_ring_cell()
    # Lower access lines are collinear, so the o1/o4 separation is
    # length_x + 2 * length_extension.
    lower_span = layout.ports["o4"].center[0] - layout.ports["o1"].center[0]
    assert lower_span == pytest.approx(length_x + 2 * length_extension)
    # Upper access lines are quarter arcs leaving the coupled section, which
    # spans x in [-length_x, 0], so each port sits radius further out.
    assert layout.ports["o3"].center[0] == pytest.approx(radius)
    assert layout.ports["o2"].center[0] == pytest.approx(-(radius + length_x))

    frequencies = jnp.linspace(4e9, 8e9, 5)
    default = coupler_ring(f=frequencies)
    explicit = coupler_ring(
        f=frequencies, radius=radius, length_extension=length_extension
    )
    for key in default:
        assert_allclose(default[key], explicit[key], rtol=1e-12, atol=1e-15)


@given(
    delta=st.floats(min_value=1.0, max_value=200.0),
    radius=st.floats(min_value=20.0, max_value=200.0),
)
@settings(max_examples=10, deadline=None)
def test_coupler_ring_length_extension_lengthens_only_lower_access(
    delta: float, radius: float
) -> None:
    """Cascade straights of length_extension onto the lower ports only."""
    frequencies = jnp.linspace(4e9, 8e9, 5)
    base = coupler_ring(f=frequencies, radius=radius, length_extension=3.0 + radius)
    extended = coupler_ring(
        f=frequencies, radius=radius, length_extension=3.0 + radius + delta
    )

    # Both lower access lines grow by delta, so the through path picks up the
    # phase of a single straight of twice that length.
    reference = straight(f=frequencies, length=2 * delta)
    assert_allclose(
        extended["o1", "o4"] / base["o1", "o4"],
        reference["o1", "o2"],
        rtol=1e-9,
        atol=1e-15,
    )
    assert_allclose(extended["o2", "o3"], base["o2", "o3"], rtol=1e-9, atol=1e-15)


@given(
    delta_radius=st.floats(min_value=1.0, max_value=100.0),
    length_extension=st.floats(min_value=1.0, max_value=50.0),
)
@settings(max_examples=10, deadline=None)
def test_coupler_ring_radius_lengthens_only_upper_access(
    delta_radius: float, length_extension: float
) -> None:
    """Cascade quarter-circle arcs of radius onto the upper ports only."""
    frequencies = jnp.linspace(4e9, 8e9, 5)
    base = coupler_ring(f=frequencies, radius=100.0, length_extension=length_extension)
    wider = coupler_ring(
        f=frequencies, radius=100.0 + delta_radius, length_extension=length_extension
    )

    # Each upper arc grows by pi * delta_radius / 2.
    reference = straight(f=frequencies, length=π * delta_radius)
    assert_allclose(
        wider["o2", "o3"] / base["o2", "o3"],
        reference["o1", "o2"],
        rtol=1e-9,
        atol=1e-15,
    )
    assert_allclose(wider["o1", "o4"], base["o1", "o4"], rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("radius", [5.0, 11.0, 100.0], ids=["below", "at", "above"])
def test_coupler_ring_clamps_bend_radius_like_layout(radius: float) -> None:
    """Clamp the upper arc to the bend cross-section's radius_min."""
    main = gf.get_cross_section("cpw")
    radius_min = main.radius_min
    assert radius_min is not None
    length_x = inspect.signature(coupler_ring).parameters["length_x"].default

    layout = coupler_ring_cell(radius=radius)
    # The lower extension follows the requested radius, the upper arc the clamp.
    lower_span = layout.ports["o4"].center[0] - layout.ports["o1"].center[0]
    assert lower_span == pytest.approx(length_x + 2 * (3.0 + radius))
    layout_radius = layout.ports["o3"].center[0]
    assert layout_radius == pytest.approx(max(radius, radius_min))

    frequencies = jnp.linspace(4e9, 8e9, 5)
    clamped = coupler_ring(f=frequencies, radius=radius, length_extension=length_x)
    reference = coupler_ring(
        f=frequencies, radius=layout_radius, length_extension=length_x
    )
    for key in clamped:
        assert_allclose(clamped[key], reference[key], rtol=1e-12, atol=1e-15)


def test_coupler_ring_cross_section_bend_reflects_at_junction() -> None:
    """Keep the impedance step where the bend cross-section has a different gap."""
    main = gf.get_cross_section("cpw")
    assert main.radius is not None
    bend = coplanar_waveguide(width=main.width, gap=3.0)
    # Equal conductor width, so the layout ports stay connectable.
    assert gf.get_cross_section(bend).width == main.width
    layout = coupler_ring_cell(cross_section_bend=bend)
    assert layout.ports["o3"].center[0] == pytest.approx(main.radius)

    frequencies = jnp.linspace(4e9, 8e9, 5)
    z_bend = cpw_z0_from_cross_section(bend)
    z_ref = cpw_z0_from_cross_section("cpw")
    rho = (z_bend - z_ref) / (z_bend + z_ref)
    assert abs(float(rho)) > 1e-3

    # A line of the bend impedance seen from the coupled section reflects by rho,
    # attenuated by its own round trip. matched is that line's e^{-gamma l}.
    length = π * main.radius / 2
    matched = straight(f=frequencies, length=length, cross_section=bend)["o1", "o2"]
    denominator = 1 - rho**2 * matched**2
    line = _access_line_s_params(
        f=frequencies, length=length, cross_section=bend, z_ref=z_ref
    )
    assert_allclose(
        line["o1", "o1"], rho * (1 - matched**2) / denominator, rtol=1e-9, atol=1e-15
    )
    assert_allclose(
        line["o1", "o2"],
        (1 - rho**2) * matched / denominator,
        rtol=1e-9,
        atol=1e-15,
    )

    model = coupler_ring(f=frequencies, cross_section_bend=bend)
    assert jnp.all(jnp.abs(model["o2", "o2"]) > 0)
    assert jnp.all(jnp.abs(model["o2", "o2"]) <= jnp.abs(rho))


@pytest.mark.parametrize("radius", [40.0, 5.0], ids=["above_min", "below_min"])
@pytest.mark.parametrize("length_extension", [None, 5.0], ids=["auto", "custom"])
def test_coupler_ring_jit_compatible(
    radius: float, length_extension: float | None
) -> None:
    """Keep the model usable inside a jitted circuit with a traced radius."""
    frequencies = jnp.linspace(4e9, 8e9, 5)
    jitted = jax.jit(
        lambda f, r: coupler_ring(f=f, radius=r, length_extension=length_extension)
    )
    expected = coupler_ring(
        f=frequencies, radius=radius, length_extension=length_extension
    )

    result = jitted(frequencies, radius)
    assert set(result) == set(expected)
    for key in expected:
        assert_allclose(result[key], expected[key], rtol=1e-12, atol=1e-15)


class TestCPWCouplingCapacitanceAnalytical:
    """Tests for cpw_cpw_coupling_capacitance_per_length_analytical."""

    @staticmethod
    def test_positive_capacitance() -> None:
        """Test that coupling capacitance per length is positive."""
        c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=5.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert float(c_pul) > 0

    @staticmethod
    def test_capacitance_decreases_with_gap() -> None:
        """Test that capacitance decreases as gap increases."""
        c_small_gap = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=1.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        c_large_gap = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=10.0, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert float(c_small_gap) > float(c_large_gap)

    @staticmethod
    def test_broadcasting() -> None:
        """Test that broadcasting works for multiple gaps."""
        gaps = jnp.geomspace(0.5, 5.0, 5)
        c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=gaps, width=10.0, cpw_gap=6.0, ep_r=11.7
        )
        assert c_pul.shape == (5,)
        assert jnp.all(c_pul > 0)


class TestCPWCouplingCapacitance:
    """Tests for cpw_cpw_coupling_capacitance."""

    @staticmethod
    def test_total_capacitance_scales_with_length() -> None:
        """Test that total coupling capacitance scales linearly with length."""
        f = jnp.array([5e9])
        c1 = cpw_cpw_coupling_capacitance(f, length=100.0, gap=5.0, cross_section="cpw")
        c2 = cpw_cpw_coupling_capacitance(f, length=200.0, gap=5.0, cross_section="cpw")
        np.testing.assert_allclose(float(c2), 2.0 * float(c1), rtol=1e-6)

    @staticmethod
    def test_missing_etch_section_raises() -> None:
        """Test that a cross-section without an etch section raises ValueError."""
        xs = gf.cross_section.cross_section(width=10.0)
        with pytest.raises(ValueError, match="etch"):
            cpw_cpw_coupling_capacitance(5e9, length=100.0, gap=5.0, cross_section=xs)
