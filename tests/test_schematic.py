"""Tests for schematic symbols."""

import importlib
from typing import Any, cast

import gdsfactory as gf
import pytest
from gdsfactory.typings import ComponentFactory
from kfactory.schematic import DSchematic

from qpdk import PDK, SAX_MODEL_ALIASES
from qpdk.cells import (
    bend_circular,
    bend_s,
    double_pad_transmon,
    launcher,
    lumped_element_resonator,
    meander_inductor,
    open as open_cell,
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
    short as short_cell,
    straight,
    straight_open,
    straight_shorted,
)
from qpdk.cells._schematic import (
    double_pad_transmon_schematic,
    straight_schematic,
)
from qpdk.models import models as sax_models


def _get_schematic(cell: Any) -> DSchematic | None:
    if schematic_function := getattr(cell, "schematic_function", None):
        return schematic_function()
    try:
        factory = gf.kcl.factories[cell.__name__]
    except (AttributeError, KeyError):
        return None
    try:
        return factory.get_schematic()
    except ValueError:
        return None


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
        s = _get_schematic(cell)
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
        open_cell: ("qpdk.models.generic", {"o1"}),
        short_cell: ("qpdk.models.generic", {"o1"}),
        straight_open: ("qpdk.models.waveguides", {"o1"}),
        straight_shorted: ("qpdk.models.waveguides", {"o1"}),
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
        schematic = _get_schematic(cell)
        assert schematic is not None

        assert set(schematic.ports) == ports
        assert schematic.info["models"][0]["module"] == module
        assert set(schematic.info["models"][0]["port_order"]) == ports

    bend_s_schematic = _get_schematic(bend_s)
    assert bend_s_schematic is not None
    bend_s_model = bend_s_schematic.info["models"][0]
    assert bend_s_model["params"] == {}


@pytest.mark.parametrize(
    "cell",
    [open_cell, short_cell, straight_open, straight_shorted],
)
def test_termination_cells_expose_one_simulation_port(
    cell: ComponentFactory,
) -> None:
    """Keep ideal and distributed terminations one-port in layout netlists."""
    component = cell()

    assert {port.name for port in component.ports.filter(port_type="optical")} == {"o1"}


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
        schematic = _get_schematic(cell)
        if schematic is None:
            continue

        for descriptor in schematic.info["models"]:
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


@pytest.mark.parametrize(("alias", "model_name"), SAX_MODEL_ALIASES.items())
def test_sax_model_aliases(alias: str, model_name: str) -> None:
    """Resolve aliased cells to the intended registered model."""
    assert PDK.models is not None
    assert alias not in sax_models
    assert model_name in sax_models
    assert alias in PDK.cells
    assert PDK.models[alias] is PDK.models[model_name]


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
