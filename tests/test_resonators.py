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
    # Ensure length for straights is positive
    bend_factory = partial(bend_circular, angle=180)
    assume(length > meanders * bend_factory().info["length"])

    c = resonator(
        length=length,
        meanders=meanders,
        open_start=open_start,
        open_end=open_end,
        bend_spec=bend_factory,
    )
    assert c is not None
    assert c.info["length"] == length
    assert len(c.ports) == 2
