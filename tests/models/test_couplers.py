"""Tests for qpdk.models.couplers module."""

from typing import final

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpdk import PDK
from qpdk.cells.waveguides import (
    bend_circular,
    coupler_ring as coupler_ring_cell,
    straight as straight_cell,
)
from qpdk.models.couplers import (
    _ring_access_lengths,
    coupler_ring,
    coupler_straight,
    cpw_cpw_coupling_capacitance,
    cpw_cpw_coupling_capacitance_per_length_analytical,
)
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.tech import LAYER, coplanar_waveguide

from .base import FourPortModelTestSuite

_RING_CASES = [
    {},
    {"radius": 60},
    {"radius": 150},
    {"length_extension": 40},
    {"radius": 60, "length_extension": 25.5},
    {"radius": 60, "length_x": 35, "gap": 12},
    # Below the cross-section minimum radius (11 µm): the bend is raised to it.
    {"radius": 5},
]
_RING_CASE_IDS = [
    "default",
    "radius60",
    "radius150",
    "extension40",
    "radius60_extension25.5",
    "radius60_lx35_gap12",
    "radius_below_min",
]


def _layout_path_lengths(**kwargs: float) -> tuple[float, float]:
    """Measure the lower and upper path lengths (o1-o4, o2-o3) of the layout in µm.

    The lower path is straight along the port axis. The upper path is the
    centre-line length of the conductor layer (area divided by width) minus
    the lower path.

    Returns:
        The lower and upper path lengths in µm.
    """
    PDK.activate()
    component = gf.c.coupler_ring(
        length_x=kwargs.pop("length_x", 20),
        gap=kwargs.pop("gap", 16),
        cross_section="cpw",
        bend=bend_circular,
        straight=straight_cell,
        **kwargs,
    )
    ports = {port.name: port.dcenter for port in component.ports}
    lower = ports["o4"][0] - ports["o1"][0]
    width, _ = get_cpw_dimensions("cpw")
    region = component.get_region(gf.get_layer(LAYER.WG)).merged()
    centre_line_total = region.area() * component.kcl.dbu**2 / width
    return lower, centre_line_total - lower


def _model_path_lengths(
    length_x: float = 20,
    gap: float = 16,  # ruff: ignore[unused-function-argument]
    radius: float | None = None,
    length_extension: float | None = None,
) -> tuple[float, float]:
    """Return the lower and upper path lengths in µm implied by the SAX model."""
    straight_length, bend_length = _ring_access_lengths("cpw", radius, length_extension)
    return (
        float(2 * straight_length + length_x),
        float(2 * bend_length + length_x),
    )


class TestCouplerRingAccessLengths:
    """The model access lengths follow the layout of the ring cell."""

    @staticmethod
    @pytest.mark.parametrize("kwargs", _RING_CASES, ids=_RING_CASE_IDS)
    def test_lengths_match_layout(kwargs: dict) -> None:
        """Lower and upper path lengths agree with the layout within 0.01 µm."""
        layout_lower, layout_upper = _layout_path_lengths(**kwargs)
        model_lower, model_upper = _model_path_lengths(**kwargs)
        assert model_lower == pytest.approx(layout_lower, abs=0.01)
        assert model_upper == pytest.approx(layout_upper, abs=0.01)

    @staticmethod
    def test_default_cell_lengths() -> None:
        """The default cell has a 226 µm lower and about 334.16 µm upper path."""
        PDK.activate()
        ports = {port.name: port.dcenter for port in coupler_ring_cell().ports}
        assert ports["o4"][0] - ports["o1"][0] == pytest.approx(226.0, abs=1e-6)
        assert _model_path_lengths() == pytest.approx((226.0, 334.1593), abs=0.01)


@final
class TestCouplerRing(FourPortModelTestSuite):
    """Tests for coupler_ring model."""

    model_function = coupler_ring

    @staticmethod
    @pytest.mark.parametrize("kwargs", _RING_CASES, ids=_RING_CASE_IDS)
    def test_path_phase_matches_added_length(kwargs: dict) -> None:
        """Each S-parameter gains the phase of the added CPW on both of its ports.

        The reference is the bare coupled section, which is the previous model.
        The added section is lossless-matched, so the phase follows
        ``-beta * (l_i + l_j)`` for ports i and j, and the magnitudes are kept.
        """
        f = jnp.linspace(4e9, 8e9, 5)
        length_x = kwargs.get("length_x", 20)
        gap = kwargs.get("gap", 16)
        straight_length, bend_length = _ring_access_lengths(
            "cpw", kwargs.get("radius"), kwargs.get("length_extension")
        )
        port_length = {  # µm
            "o1": float(straight_length),
            "o4": float(straight_length),
            "o2": float(bend_length),
            "o3": float(bend_length),
        }
        width, cpw_gap = get_cpw_dimensions("cpw")
        ep_eff, _ = cpw_parameters(width, cpw_gap, tand=0.0)
        beta = 2 * np.pi * np.asarray(f) * np.sqrt(float(ep_eff.real)) / 299792458.0

        ring = coupler_ring(f=f, **kwargs)
        bare = coupler_straight(f=f, length=length_x, gap=gap, cross_section="cpw")
        for (port_i, port_j), reference in bare.items():
            added = beta * (port_length[port_i] + port_length[port_j]) * 1e-6
            shift = np.asarray(ring[port_i, port_j] / reference) * np.exp(1j * added)
            np.testing.assert_allclose(np.angle(shift), 0.0, atol=1e-4)
            np.testing.assert_allclose(
                np.abs(ring[port_i, port_j]), np.abs(reference), rtol=1e-3, atol=1e-9
            )

    @staticmethod
    def test_geometry_changes_response() -> None:
        """``radius`` and ``length_extension`` change the model like the layout."""
        f = jnp.array([6e9])
        baseline = coupler_ring(f=f)
        # The default extension is 3 + radius, so a radius change moves both paths.
        smaller = coupler_ring(f=f, radius=60)
        assert not jnp.allclose(smaller["o2", "o3"], baseline["o2", "o3"])
        assert not jnp.allclose(smaller["o1", "o4"], baseline["o1", "o4"])
        # An explicit extension moves the lower path only.
        extended = coupler_ring(f=f, length_extension=40)
        assert not jnp.allclose(extended["o1", "o4"], baseline["o1", "o4"])
        assert jnp.allclose(extended["o2", "o3"], baseline["o2", "o3"])

    @staticmethod
    def test_jit_matches_eager() -> None:
        """The model is jittable, with traced geometry parameters."""
        f = jnp.linspace(4e9, 8e9, 5)
        compiled = jax.jit(lambda r, e: coupler_ring(f=f, radius=r, length_extension=e))
        result = compiled(60.0, 25.5)
        expected = coupler_ring(f=f, radius=60.0, length_extension=25.5)
        for key, value in expected.items():
            np.testing.assert_allclose(result[key], value, atol=1e-6)

    @staticmethod
    def test_phase_gradient_matches_electrical_length() -> None:
        """d(phase)/d(geometry) equals the phase constant times the length slope."""
        f = jnp.array([6e9])
        width, cpw_gap = get_cpw_dimensions("cpw")
        ep_eff, _ = cpw_parameters(width, cpw_gap, tand=0.0)
        beta = 2 * np.pi * 6e9 * np.sqrt(float(ep_eff.real)) / 299792458.0
        beta_per_um = beta * 1e-6

        d_upper = jax.grad(
            lambda r: jnp.angle(coupler_ring(f=f, radius=r)["o2", "o3"][0])
        )(100.0)
        d_lower = jax.grad(
            lambda e: jnp.angle(coupler_ring(f=f, length_extension=e)["o1", "o4"][0])
        )(103.0)
        # Upper path: 2 bends of pi * radius / 2; lower path: 2 straights.
        np.testing.assert_allclose(d_upper, -beta_per_um * np.pi, rtol=1e-3)
        np.testing.assert_allclose(d_lower, -beta_per_um * 2, rtol=1e-3)


@final
class TestCouplerRingNonDefaultGeometry(FourPortModelTestSuite):
    """Reciprocity and passivity with non-default ring geometry."""

    model_function = coupler_ring

    @staticmethod
    def get_model_kwargs() -> dict:
        """Use a smaller radius and a custom extension."""
        return {"radius": 60.0, "length_extension": 25.5}


@final
class TestCouplerStraight(FourPortModelTestSuite):
    """Tests for coupler_straight model."""

    model_function = coupler_straight

    @staticmethod
    def test_upper_bus_cross_section_changes_response() -> None:
        """The bus override changes the upper path while keeping the resonator path."""
        frequencies = jnp.array([5e9, 6e9])
        baseline = coupler_straight(f=frequencies, length=200)
        modified = coupler_straight(
            f=frequencies,
            length=200,
            cross_section_non_resonator=coplanar_waveguide(width=20, gap=10),
        )
        assert not jnp.allclose(modified["o3", "o2"], baseline["o3", "o2"])


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
