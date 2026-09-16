"""Tests for schematic symbols."""

import importlib
from typing import Any, cast

from kfactory.schematic import DSchematic

from qpdk import PDK
from qpdk.cells import (
    bend_circular,
    bend_s,
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
        bend_s,
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
        straight: ("qpdk.models.waveguides", {"o1", "o2"}),
        bend_s: ("qpdk.models.waveguides", {"o1", "o2"}),
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

    bend_s_model = cast(Any, bend_s).schematic_function().info["models"][0]
    assert bend_s_model["params"] == {}


def test_sax_model_descriptors_resolve_from_the_pdk_registry() -> None:
    """Keep every declared model importable, registered, and port-compatible."""
    required_fields = {
        "language",
        "name",
        "module",
        "qualname",
        "port_order",
        "params",
    }
    seen_cells: set[int] = set()
    descriptor_count = 0

    for cell in PDK.cells.values():
        if id(cell) in seen_cells:
            continue
        seen_cells.add(id(cell))
        schematic_function = getattr(cell, "schematic_function", None)
        if schematic_function is None:
            continue

        for descriptor in schematic_function().info["models"]:
            descriptor_count += 1
            assert required_fields <= descriptor.keys()
            assert descriptor["language"] == "sax"

            module = importlib.import_module(descriptor["module"])
            model = getattr(module, descriptor["qualname"])
            assert PDK.models[descriptor["name"]] is model

            s_params = model(f=[7e9])
            model_ports = {port for pair in s_params for port in pair}
            assert model_ports == set(descriptor["port_order"])

    assert descriptor_count > 0


def test_resonator_variants_have_direct_sax_models() -> None:
    """Resolve bend variants through their declared model metadata."""
    assert PDK.models is not None

    for wave_type in ("quarter_wave", "half_wave"):
        for bend_type in ("start", "end", "both"):
            name = f"resonator_{wave_type}_bend_{bend_type}"
            cell = PDK.cells[name]
            schematic = cast(Any, cell).schematic_function()
            model_name = schematic.info["models"][0]["name"]
            s_params = PDK.models[model_name](f=[7e9])

            assert {port for key in s_params for port in key} == {"o1", "o2"}


def test_coupled_resonators_declare_model_boundaries() -> None:
    """Stop hierarchy expansion where capacitive coupling is modeled."""
    for cell in (resonator_coupled, quarter_wave_resonator_coupled):
        schematic = cast(Any, cell).schematic_function()

        assert schematic.info["models"][0]["name"] == cell.__name__


def test_test_chip_does_not_declare_a_whole_chip_model() -> None:
    """Allow chip assemblies to expand until modeled component boundaries."""
    cell = PDK.cells["resonator_test_chip_python"]
    schematic = cast(Any, cell).schematic_function()

    assert schematic.info["models"] == []
