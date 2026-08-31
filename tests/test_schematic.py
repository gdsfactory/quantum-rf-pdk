"""Tests for schematic symbols."""

from typing import Any, cast

from kfactory.schematic import DSchematic

from qpdk import PDK
from qpdk.cells import (
    bend_circular,
    double_pad_transmon,
    launcher,
    lumped_element_resonator,
    meander_inductor,
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_half_wave,
    resonator_half_wave_bend_both,
    resonator_half_wave_bend_end,
    resonator_half_wave_bend_start,
    resonator_quarter_wave,
    resonator_quarter_wave_bend_both,
    resonator_quarter_wave_bend_end,
    resonator_quarter_wave_bend_start,
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
        resonator_half_wave,
        resonator_half_wave_bend_start,
        resonator_half_wave_bend_end,
        resonator_half_wave_bend_both,
        resonator_quarter_wave,
        resonator_quarter_wave_bend_start,
        resonator_quarter_wave_bend_end,
        resonator_quarter_wave_bend_both,
        resonator_coupled,
        quarter_wave_resonator_coupled,
        launcher,
        lumped_element_resonator,
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


def test_simulation_cells_have_sax_models() -> None:
    """Expose SAX-backed symbols with the same ports as their layout cells."""
    expected = {
        launcher: ("qpdk.models.waveguides", {"waveport", "o1"}),
        lumped_element_resonator: ("qpdk.models.inductor", {"o1", "o2"}),
        resonator: ("qpdk.models.resonator", {"o1", "o2"}),
        resonator_half_wave: ("qpdk.models.resonator", {"o1", "o2"}),
        resonator_quarter_wave: ("qpdk.models.resonator", {"o1", "o2"}),
        resonator_coupled: (
            "qpdk.models.resonator",
            {"coupling_o1", "coupling_o2", "resonator_o1", "resonator_o2"},
        ),
        quarter_wave_resonator_coupled: (
            "qpdk.models.resonator",
            {"coupling_o1", "coupling_o2", "resonator_o1"},
        ),
    }

    for cell, (module, ports) in expected.items():
        schematic = cast(Any, cell).schematic_function()

        assert set(schematic.ports) == ports
        assert schematic.info["models"][0]["module"] == module
        assert set(schematic.info["models"][0]["port_order"]) == ports


def test_resonator_variants_have_direct_sax_models() -> None:
    """Make every registered bend variant directly simulatable."""
    assert PDK.models is not None

    for wave_type in ("quarter_wave", "half_wave"):
        for bend_type in ("start", "end", "both"):
            name = f"resonator_{wave_type}_bend_{bend_type}"
            s_params = PDK.models[name](f=[7e9])

            assert {port for key in s_params for port in key} == {"o1", "o2"}
