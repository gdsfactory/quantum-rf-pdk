"""Test resonator generation and properties."""

from functools import partial

import hypothesis.strategies as st
from hypothesis import assume, given, settings

from qpdk.cells.resonator import resonator_quarter_wave
from qpdk.cells.waveguides import bend_circular


@given(
    length=st.floats(min_value=0, max_value=1000000),
    meanders=st.integers(min_value=1),
)
@settings(max_examples=50, deadline=None)
def test_resonator_meanders(length: float, meanders: int) -> None:
    # Ensure length for straights is positive
    bend_factory = partial(bend_circular, angle=180)
    assume(length > meanders * bend_factory().info["length"])

    c = resonator_quarter_wave(length=length, meanders=meanders, bend_spec=bend_factory)
    assert c is not None
    assert c.info["length"] == length
    assert c is not None
    assert c.info["length"] == length
