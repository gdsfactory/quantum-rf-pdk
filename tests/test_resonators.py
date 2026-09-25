"""Test resonator generation and properties."""

import inspect
import math
from functools import partial
from typing import Any

import gdsfactory as gf
import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings

from qpdk.cells.derived.transmon_with_resonator_and_probeline import (
    double_pad_transmon_with_resonator,
    flipmon_with_resonator,
)
from qpdk.cells.inductor import meander_inductor
from qpdk.cells.resonator import (
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_half_wave,
    resonator_quarter_wave,
)
from qpdk.cells.waveguides import bend_circular
from qpdk.tech import LAYER, get_etch_section

MAX_EXAMPLES = 20


def _offset_point(port: Any, distance: float) -> tuple[float, float]:
    """Point at distance from a port center along its orientation."""
    angle = math.radians(port.orientation)
    return (
        port.center[0] + distance * math.cos(angle),
        port.center[1] + distance * math.sin(angle),
    )


def _etch_covers(component: gf.Component, point: tuple[float, float]) -> bool:
    """Whether the merged M1_ETCH region covers a point given in um."""
    # kdb points are integers in database units, 1 nm for this PDK
    probe = gf.kdb.Point(round(point[0] * 1000), round(point[1] * 1000))
    region = gf.kdb.Region(
        component.begin_shapes_rec(gf.get_layer(LAYER.M1_ETCH))
    ).merged()
    return any(polygon.inside(probe) for polygon in region.each())


class TestResonators:
    """Test basic resonator functionality."""

    @staticmethod
    def test_cross_section_info_does_not_shadow_settings() -> None:
        """Keep physical metadata separate from SAX model settings."""
        resonator_component = resonator(cross_section="cpw")
        inductor_component = meander_inductor(cross_section="cpw")

        assert resonator_component.info["cross_section_name"] == "coplanar_waveguide"
        assert inductor_component.info["cross_section_name"] == "coplanar_waveguide"
        assert "cross_section" not in resonator_component.info
        assert "cross_section" not in inductor_component.info

    @staticmethod
    def test_resonator_type_metadata_matches_termination() -> None:
        """Distinguish half-wave and quarter-wave layouts in metadata."""
        assert resonator_half_wave().info["resonator_type"] == "half_wave"
        assert resonator_quarter_wave().info["resonator_type"] == "quarter_wave"

    @staticmethod
    @pytest.mark.parametrize("length", [0, 10, 100])
    def test_resonator_too_short_raises_error(length: float) -> None:
        """Verify that providing a length too short for the meanders raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            resonator(length=length, meanders=6)

    @staticmethod
    @given(
        length=st.floats(min_value=0, max_value=1000000),
        meanders=st.integers(min_value=1, max_value=100),
        open_start=st.booleans(),
        open_end=st.booleans(),
        start_with_bend=st.booleans(),
        end_with_bend=st.booleans(),
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_resonator_meanders(
        length: float,
        meanders: int,
        open_start: bool,
        open_end: bool,
        start_with_bend: bool,
        end_with_bend: bool,
    ) -> None:
        bend_factory = partial(bend_circular, angle=180, angular_step=4)

        # Ensure total length is sufficient to accommodate all bends
        # Each meander requires space for the bend sections
        bend_length = bend_factory().info["length"]

        num_straights = meanders + 1
        if start_with_bend:
            num_straights -= 1
        if end_with_bend:
            num_straights -= 1

        if num_straights > 0:
            assume(length > meanders * bend_length)
        else:
            # If no straights, the length must be exactly meanders * bend_length
            # for consistency, but the component creation should still work.
            # However, the current implementation of resonator() just uses 'length'
            # to calculate straights length IF num_straights > 0.
            # If num_straights == 0, 'length' is just stored in metadata.
            pass

        c = resonator(
            length=length,
            meanders=meanders,
            open_start=open_start,
            open_end=open_end,
            start_with_bend=start_with_bend,
            end_with_bend=end_with_bend,
            bend_spec=bend_factory,
        )

        expected_length = length if num_straights > 0 else meanders * bend_length

        assert c is not None, "Resonator component should be created successfully"
        assert c.info["length"] == pytest.approx(expected_length), (
            f"Expected length {expected_length}, got {c.info['length']}"
        )
        assert len(c.ports) == 2, f"Expected 2 ports, got {len(c.ports)}"

    @staticmethod
    @given(
        length=st.floats(min_value=0, max_value=1000000),
        meanders=st.integers(min_value=1, max_value=100),
        open_start=st.booleans(),
        open_end=st.booleans(),
        start_with_bend=st.booleans(),
        end_with_bend=st.booleans(),
        coupling_straight_length=st.floats(min_value=1, max_value=1000),
        coupling_gap=st.floats(min_value=1, max_value=100),
    )
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_resonator_coupled(
        length: float,
        meanders: int,
        open_start: bool,
        open_end: bool,
        start_with_bend: bool,
        end_with_bend: bool,
        coupling_straight_length: float,
        coupling_gap: float,
    ) -> None:
        bend_factory = partial(bend_circular, angle=180, angular_step=4)

        # Ensure total length is sufficient to accommodate all bends
        # Each meander requires space for the bend sections
        bend_length = bend_factory().info["length"]

        num_straights = meanders + 1
        if start_with_bend:
            num_straights -= 1
        if end_with_bend:
            num_straights -= 1

        if num_straights > 0:
            assume(length > meanders * bend_length)

        c = resonator_coupled(
            length=length,
            meanders=meanders,
            open_start=open_start,
            open_end=open_end,
            start_with_bend=start_with_bend,
            end_with_bend=end_with_bend,
            bend_spec=bend_factory,
            coupling_straight_length=coupling_straight_length,
            coupling_gap=coupling_gap,
        )

        expected_length = length if num_straights > 0 else meanders * bend_length

        assert c is not None, (
            "Coupled resonator component should be created successfully"
        )
        assert c.info["length"] == pytest.approx(expected_length), (
            f"Expected length {expected_length}, got {c.info['length']}"
        )
        assert c.info["coupling_length"] == coupling_straight_length, (
            f"Expected coupling length {coupling_straight_length}, got {c.info['coupling_length']}"
        )
        assert c.info["coupling_gap"] == coupling_gap, (
            f"Expected coupling gap {coupling_gap}, got {c.info['coupling_gap']}"
        )
        # Should have 4 ports: 2 from resonator + 2 from coupling waveguide
        assert len(c.ports) == 4, f"Expected 4 ports, got {len(c.ports)}"

        # Check that we have the expected port names
        port_names = [p.name for p in c.ports]
        expected_ports = {"resonator_o1", "resonator_o2", "coupling_o1", "coupling_o2"}
        assert set(port_names) == expected_ports, (
            f"Expected ports {expected_ports}, got {set(port_names)}"
        )


class TestQuarterWaveResonatorCoupled:
    """Test the fixed three-port quarter-wave coupled resonator."""

    @staticmethod
    def test_terminations_are_not_public_settings() -> None:
        """Terminations are fixed, so the layout factory hides them."""
        parameters = inspect.signature(quarter_wave_resonator_coupled).parameters

        assert "open_start" not in parameters
        assert "open_end" not in parameters

    @staticmethod
    def test_nested_coupled_resonator_terminations() -> None:
        """The coupled end is open and the hidden far end stays shorted."""
        c = quarter_wave_resonator_coupled()

        assert {port.name for port in c.ports} == {
            "coupling_o1",
            "coupling_o2",
            "resonator_o1",
        }

        netlist = c.get_netlist()
        (instance,) = netlist["instances"].values()

        assert instance["component"] == "resonator_coupled"
        assert instance["settings"]["open_start"] is True
        assert instance["settings"]["open_end"] is False
        assert c.info["resonator_type"] == "quarter_wave"

        # Pin the etch geometry, not just the settings literals above: an
        # open end is capped by an etch patch past the trace end, a shorted
        # end is bare. The patch extends one etch-section width outwards.
        etch_width = get_etch_section(gf.get_cross_section("cpw")).width
        port = c.ports["resonator_o1"]
        assert _etch_covers(c, _offset_point(port, -etch_width / 2))

        # The hidden far end sits where resonator_coupled's shorted port is,
        # relative to the coupling port the wrapper normalizes to the origin.
        inner = resonator_coupled()
        far_port = inner["resonator_o2"]
        offset = inner["coupling_o1"].center
        far_point = _offset_point(far_port, etch_width / 2)
        far_point = (far_point[0] - offset[0], far_point[1] - offset[1])
        assert not _etch_covers(c, far_point)


class TestQubitWithResonator:
    """Test qubit—resonator coupled systems."""

    @staticmethod
    def test_transmon_with_resonator_defaults() -> None:
        """Test transmon_with_resonator with default parameters."""
        c = double_pad_transmon_with_resonator()

        assert c is not None, (
            "Transmon-resonator component should be created successfully"
        )
        assert "qubit_type" in c.info, "Component should have qubit_type info"
        assert "resonator_type" in c.info, "Component should have resonator_type info"
        assert "coupler_type" in c.info, "Component should have coupler_type info"
        assert "length" in c.info, "Component should have length info"

        # Check expected ports
        port_names = [p.name for p in c.ports]
        assert "junction" in port_names, "Should have junction port from transmon"
        assert "o1" in port_names, "Should have o1 port from resonator"

    @staticmethod
    def test_flipmon_with_resonator_defaults() -> None:
        """Test flipmon_with_resonator with default parameters."""
        c = flipmon_with_resonator()

        assert c is not None, (
            "Flipmon-resonator component should be created successfully"
        )
        assert "qubit_type" in c.info, "Component should have qubit_type info"
        assert "resonator_type" in c.info, "Component should have resonator_type info"
        assert "coupler_type" in c.info, "Component should have coupler_type info"
        assert "length" in c.info, "Component should have length info"

        # Check expected ports
        port_names = [p.name for p in c.ports]
        assert "junction" in port_names, "Should have junction port from flipmon"
        assert "o1" in port_names, "Should have o1 port from resonator"
