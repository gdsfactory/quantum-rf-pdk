"""Test resonator generation and properties."""

from functools import partial

import hypothesis.strategies as st
from hypothesis import assume, given, settings

from qpdk.cells import resonator
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
