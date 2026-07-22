"""Tests for schematic symbols."""

from kfactory.schematic import DSchematic

from qpdk.cells import (
    bend_circular,
    double_pad_transmon,
    meander_inductor,
    resonator,
    straight,
)
from qpdk.cells._schematic import (
    double_pad_transmon_schematic,
    straight_schematic,
)


def test_schematic_functions():
    """Verify that schematic functions are attached to cells."""
    cells = [
        straight,
        bend_circular,
        resonator,
        double_pad_transmon,
        meander_inductor,
    ]

    for cell in cells:
        # Check if schematic_function is attached to the cell
        assert hasattr(cell, "schematic_function")
        assert cell.schematic_function is not None

        # Execute it and verify it returns a DSchematic
        s = cell.schematic_function()
        assert isinstance(s, DSchematic)
        assert "symbol" in s.info


def test_schematic_factory():
    """Verify that schematic factory returns correct DSchematic objects."""
    s = straight_schematic()
    assert isinstance(s, DSchematic)
    assert s.info["symbol"] == "straight"
    assert "o1" in s.ports
    assert "o2" in s.ports

    s = double_pad_transmon_schematic()
    assert isinstance(s, DSchematic)
    assert s.info["symbol"] == "double_pad_transmon"
    assert "left_pad" in s.ports
    assert "right_pad" in s.ports
