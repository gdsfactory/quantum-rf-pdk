"""Test resonator generation and properties."""

from functools import partial

import hypothesis.strategies as st
from hypothesis import assume, given, settings

from qpdk.cells import resonator
from qpdk.cells.resonator import inductor, lumped_resonator
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
    length=st.floats(min_value=1000, max_value=10000),
    meanders=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50, deadline=None)
def test_inductor_basic(length: float, meanders: int) -> None:
    """Test basic inductor creation with various parameters."""
    # Ensure inductor has enough length for the meanders
    # Each bend is about 314 μm (π * 100 μm radius)
    min_length = meanders * 320  # Safety margin
    assume(length > min_length)

    c = inductor(
        length=length,
        meanders=meanders,
    )

    assert c is not None, "Inductor component should be created successfully"
    assert c.info["length"] == length, f"Expected length {length}, got {c.info['length']}"
    assert c.info["meanders"] == meanders, f"Expected {meanders} meanders, got {c.info['meanders']}"
    assert len(c.ports) == 2, f"Expected 2 ports, got {len(c.ports)}"
    assert c.info["inductor_type"] == "meandering"


@given(
    inductor_length=st.floats(min_value=1000, max_value=5000),
    inductor_meanders=st.integers(min_value=1, max_value=5),
    capacitor_fingers=st.integers(min_value=2, max_value=20),
    capacitor_finger_length=st.floats(min_value=10, max_value=100),
)
@settings(max_examples=50, deadline=None)
def test_lumped_resonator_basic(
    inductor_length: float,
    inductor_meanders: int,
    capacitor_fingers: int,
    capacitor_finger_length: float,
) -> None:
    """Test basic lumped resonator creation with various parameters."""
    # Ensure inductor has enough length for the meanders
    min_length = inductor_meanders * 320  # Safety margin
    assume(inductor_length > min_length)

    c = lumped_resonator(
        inductor_length=inductor_length,
        inductor_meanders=inductor_meanders,
        capacitor_fingers=capacitor_fingers,
        capacitor_finger_length=capacitor_finger_length,
    )

    assert c is not None, "Lumped resonator component should be created successfully"
    assert c.info["inductor_length"] == inductor_length
    assert c.info["capacitor_fingers"] == capacitor_fingers
    assert c.info["resonator_type"] == "lumped_element"
    assert len(c.ports) == 2, f"Expected 2 ports, got {len(c.ports)}"


def test_inductor_default_params() -> None:
    """Test inductor with default parameters."""
    c = inductor()
    assert c is not None
    assert c.info["length"] == 2000.0
    assert c.info["meanders"] == 4
    assert len(c.ports) == 2


def test_lumped_resonator_default_params() -> None:
    """Test lumped resonator with default parameters."""
    c = lumped_resonator()
    assert c is not None
    assert c.info["inductor_length"] == 1500.0
    assert c.info["capacitor_fingers"] == 6
    assert c.info["resonator_type"] == "lumped_element"
    assert len(c.ports) == 2
