"""Test resonator generation and properties."""

from functools import partial

import hypothesis.strategies as st
from hypothesis import assume, given, settings

from qpdk.cells.resonator import ResonatorParams, resonator, resonator_coupled
from qpdk.cells.waveguides import bend_circular


@given(
    length=st.floats(min_value=0, max_value=1000000),
    meanders=st.integers(min_value=1),
    open_start=st.booleans(),
    open_end=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_resonator_meanders(
    length: float, meanders: int, open_start: bool, open_end: bool
) -> None:
    bend_factory = partial(bend_circular, angle=180)

    # Ensure total length is sufficient to accommodate all bends
    # Each meander requires space for the bend sections
    assume(length > meanders * bend_factory().info["length"])

    c = resonator(
        length=length,
        meanders=meanders,
        open_start=open_start,
        open_end=open_end,
        bend_spec=bend_factory,
    )

    assert c is not None, "Resonator component should be created successfully"
    assert c.info["length"] == length, (
        f"Expected length {length}, got {c.info['length']}"
    )
    assert len(c.ports) == 2, f"Expected 2 ports, got {len(c.ports)}"


@given(
    length=st.floats(min_value=0, max_value=1000000),
    meanders=st.integers(min_value=1),
    open_start=st.booleans(),
    open_end=st.booleans(),
    coupling_straight_length=st.floats(min_value=1, max_value=1000),
    coupling_gap=st.floats(min_value=1, max_value=100),
)
@settings(max_examples=50, deadline=None)
def test_resonator_coupled(
    length: float,
    meanders: int,
    open_start: bool,
    open_end: bool,
    coupling_straight_length: float,
    coupling_gap: float,
) -> None:
    bend_factory = partial(bend_circular, angle=180)

    # Ensure total length is sufficient to accommodate all bends
    # Each meander requires space for the bend sections
    assume(length > meanders * bend_factory().info["length"])

    c = resonator_coupled(
        ResonatorParams(
            length=length,
            meanders=meanders,
            open_start=open_start,
            open_end=open_end,
            bend_spec=bend_factory,
        ),
        coupling_straight_length=coupling_straight_length,
        coupling_gap=coupling_gap,
    )

    assert c is not None, "Coupled resonator component should be created successfully"
    assert c.info["length"] == length, (
        f"Expected length {length}, got {c.info['length']}"
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
