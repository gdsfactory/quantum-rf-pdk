"""Tests for qpdk.models.couplers module."""

import inspect
from typing import final

import gdsfactory as gf
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from gdsfactory.typings import CrossSectionSpec
from hypothesis import given, settings
from numpy.testing import assert_allclose
from sax.models.rf import (
    cpw_epsilon_eff,
    cpw_thickness_correction,
    propagation_constant,
)

from qpdk.cells.waveguides import coupler_ring as coupler_ring_cell
from qpdk.logger import logger
from qpdk.models.constants import π
from qpdk.models.couplers import (
    _access_line_s_params,
    coupler_ring,
    coupler_straight,
    cpw_cpw_coupling_capacitance,
    cpw_cpw_coupling_capacitance_per_length_analytical,
)
from qpdk.models.cpw import (
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
    get_cpw_substrate_params,
)
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
class TestCouplerRingBendCrossSection(FourPortModelTestSuite):
    """Shared port, reciprocity, and passivity checks with a stepped bend."""

    model_function = coupler_ring

    @staticmethod
    def get_model_kwargs() -> dict:
        """Put the junction impedance step inside the shared model checks."""
        main = gf.get_cross_section("cpw")
        return {"cross_section_bend": coplanar_waveguide(width=main.width, gap=3.0)}


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
    # The arc uses the clamped radius while the lower extension keeps being
    # derived from the requested, unclamped one.
    clamped = coupler_ring(f=frequencies, radius=radius)
    reference = coupler_ring(
        f=frequencies, radius=layout_radius, length_extension=3.0 + radius
    )
    for key in clamped:
        assert_allclose(clamped[key], reference[key], rtol=1e-12, atol=1e-15)


def test_coupler_ring_warns_when_clamping_bend_radius() -> None:
    """Warn like the layout when the clamp engages, but not under tracing."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        frequencies = jnp.linspace(4e9, 8e9, 5)
        coupler_ring(f=frequencies, radius=5.0)
        jax.jit(lambda r: coupler_ring(f=frequencies, radius=r))(5.0)
    finally:
        logger.remove(handler_id)
    assert sum("Bend radius needs to be" in str(m) for m in messages) == 1


def _mixed_reference_line(
    frequencies: jnp.ndarray,
    length: float,
    cross_section: CrossSectionSpec,
    z_ref: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Independently assemble the mixed-reference access line from its ABCD matrix.

    A uniform line of the access cross-section's impedance *z0* with port
    ``o2`` referenced to *z_ref* (the coupled section) and port ``o1`` to *z0*
    (the outer layout port sits on the access cross-section).

    Returns:
        S-parameters keyed by port pairs, matching ``_access_line_s_params``.
    """
    width, gap = get_cpw_dimensions(cross_section)
    h, t, ep_r, tand = get_cpw_substrate_params()
    ep_eff = cpw_epsilon_eff(width * 1e-6, gap * 1e-6, h * 1e-6, ep_r)
    ep_eff, z0 = cpw_thickness_correction(width * 1e-6, gap * 1e-6, t * 1e-6, ep_eff)
    z0 = float(z0)
    gamma = np.asarray(
        propagation_constant(jnp.asarray(frequencies), ep_eff, tand=tand, ep_r=ep_r)
    )
    theta = gamma * length * 1e-6
    a, b = np.cosh(theta), z0 * np.sinh(theta)
    c, d = np.sinh(theta) / z0, np.cosh(theta)
    z01, z02 = z_ref, z0  # port 1 is o2, port 2 is o1
    denom = a * z02 + b + c * z01 * z02 + d * z01
    s11 = (a * z02 + b - c * z01 * z02 - d * z01) / denom
    s22 = (d * z01 + b - c * z01 * z02 - a * z02) / denom
    thru = 2 * np.sqrt(z01 * z02) / denom
    return {
        ("o2", "o2"): s11,
        ("o1", "o1"): s22,
        ("o1", "o2"): thru,
        ("o2", "o1"): thru,
    }


def test_coupler_ring_cross_section_bend_reflects_at_junction() -> None:
    """Keep the impedance step where the bend cross-section has a different gap.

    The layout ports o2/o3 sit on the bend cross-section, so each arc has one
    impedance step, at its coupler-facing end. Renormalizing the outer port too
    would plant a second, fictitious step at the reference plane.
    """
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

    length = π * main.radius / 2
    line = _access_line_s_params(
        f=frequencies, length=length, cross_section=bend, z_ref=z_ref
    )
    # One interface: seen from the coupled section it sits at distance zero, so
    # the reflection is flat rho; seen from the outer port it is the same
    # interface one matched round trip away.
    matched = straight(f=frequencies, length=length, cross_section=bend)["o1", "o2"]
    assert_allclose(line["o2", "o2"], rho, rtol=1e-12, atol=1e-15)
    assert_allclose(line["o1", "o1"], -rho * matched**2, rtol=1e-9, atol=1e-15)
    assert_allclose(
        line["o1", "o2"], np.sqrt(1 - rho**2) * matched, rtol=1e-9, atol=1e-15
    )
    # The same network assembled independently from its ABCD matrix.
    independent = _mixed_reference_line(frequencies, length, bend, float(z_ref))
    for key, value in independent.items():
        assert_allclose(line[key], value, rtol=1e-9, atol=1e-15)

    # The whole ring is the coupler with lower straights and exactly these
    # mixed-reference arcs, so the wiring is pinned at ring level too.
    length_x = inspect.signature(coupler_ring).parameters["length_x"].default
    gap = inspect.signature(coupler_ring).parameters["gap"].default
    arc = _mixed_reference_line(frequencies, length, bend, float(z_ref))
    assembled = sax.evaluate_circuit_fg(
        (
            {
                "lower_left,o2": "coupler,o1",
                "upper_left,o2": "coupler,o2",
                "upper_right,o2": "coupler,o3",
                "lower_right,o1": "coupler,o4",
            },
            {
                "o1": "lower_left,o1",
                "o2": "upper_left,o1",
                "o3": "upper_right,o1",
                "o4": "lower_right,o2",
            },
        ),
        {
            "coupler": coupler_straight(
                f=frequencies, length=length_x, gap=gap, cross_section="cpw"
            ),
            "lower_left": straight(
                f=frequencies, length=3.0 + main.radius, cross_section="cpw"
            ),
            "lower_right": straight(
                f=frequencies, length=3.0 + main.radius, cross_section="cpw"
            ),
            "upper_left": arc,
            "upper_right": arc,
        },
    )
    model = coupler_ring(f=frequencies, cross_section_bend=bend)
    for key in model:
        assert_allclose(model[key], assembled[key], rtol=1e-9, atol=1e-15)


def test_access_line_cascade_does_not_add_reflection() -> None:
    """Extending the arc must not change what the coupled section sees.

    The outer port is referenced to the access line's own impedance, so
    attaching more of the same line leaves exactly the junction reflection.
    """
    main = gf.get_cross_section("cpw")
    assert main.radius is not None
    bend = coplanar_waveguide(width=main.width, gap=3.0)
    frequencies = jnp.linspace(4e9, 8e9, 5)
    z_bend = cpw_z0_from_cross_section(bend)
    z_ref = cpw_z0_from_cross_section("cpw")
    rho = float((z_bend - z_ref) / (z_bend + z_ref))

    arc = _access_line_s_params(
        f=frequencies, length=π * main.radius / 2, cross_section=bend, z_ref=z_ref
    )
    tail = straight(f=frequencies, length=123.0, cross_section=bend)
    composite = sax.evaluate_circuit_fg(
        ({"arc,o1": "tail,o1"}, {"coupler": "arc,o2", "outer": "tail,o2"}),
        {"arc": arc, "tail": tail},
    )
    assert_allclose(composite["coupler", "coupler"], rho, rtol=1e-9, atol=1e-12)


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
