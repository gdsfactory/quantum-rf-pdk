"""Tests for schematic symbols."""

from __future__ import annotations

import importlib
import re
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import gdsfactory as gf
import pytest
from gdsfactory.typings import ComponentFactory
from kfactory.schematic import DSchematic, Port

from qpdk import PDK, SAX_MODEL_ALIASES, _get_cell_schematic
from qpdk.cells import (
    bend_circular,
    bend_s,
    double_pad_transmon,
    double_pad_transmon_with_bbox,
    fluxonium,
    fluxonium_with_bbox,
    interdigital_capacitor,
    interdigital_capacitor_half,
    josephson_junction,
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
    unimon_coupled,
)
from qpdk.cells._schematic import (
    double_pad_transmon_schematic,
    straight_schematic,
)
from qpdk.models import models as sax_models

if TYPE_CHECKING:
    from kfactory.schematic import TSchematic


def _info(schematic: TSchematic[Any], key: str) -> Any:
    """Return a free-form ``info`` entry; the tests assert its JSON shape."""
    return schematic.info[key]


def _get_schematic(cell: Any) -> TSchematic[Any] | None:
    function = cell.func if isinstance(cell, partial) else cell
    return _get_cell_schematic(function.__name__, cell)


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
        double_pad_transmon_with_bbox,
        fluxonium,
        fluxonium_with_bbox,
        josephson_junction,
        unimon_coupled,
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
    assert s.info["symbol"] == "ckt"
    assert s.info["display"] == {"symbol": "qpdk.symbols:straight.svg"}
    assert "o1" in s.ports
    assert "o2" in s.ports

    s = double_pad_transmon_schematic()
    assert isinstance(s, DSchematic)
    assert s.info["symbol"] == "ckt"
    assert s.info["display"] == {"symbol": "qpdk.symbols:double_pad_transmon.svg"}
    assert "left_pad" in s.ports
    assert "right_pad" in s.ports


_SIDES = ("left", "right", "top", "bottom")


def _mosaic_box(ports: list[dict]) -> tuple[int, int]:
    """Grid size of the GDSFactory+ schematic symbol box for ``ports``.

    Mirrors ``port-perimeter`` in GDSFactory+'s Mosaic editor (``common.cljc``),
    which scales the SVG into this box.

    Returns:
        ``(width, height)`` of the box in grid units.
    """
    n = {side: sum(p["side"] == side for p in ports) for side in _SIDES}
    height = max(1, n["left"], n["right"])
    if n["left"] % 2 != n["right"] % 2 and height % 2 == 0:
        height += 1
    width = max(1, n["top"], n["bottom"])
    if n["top"] % 2 != n["bottom"] % 2 and width % 2 == 0:
        width += 1
    return width, height


def _pdk_schematics() -> dict[str, TSchematic[Any]]:
    schematics = {}
    for name, cell in PDK.cells.items():
        if (schematic := _get_schematic(cell)) is not None:
            schematics[name] = schematic
    return schematics


def test_schematics_use_packaged_symbols() -> None:
    """Draw every schematic as a GDSFactory+ ``ckt`` with fitting artwork.

    Any other ``symbol`` value makes GDSFactory+ show an empty box.
    """
    referenced = set()
    for name, schematic in _pdk_schematics().items():
        assert schematic.info["symbol"] == "ckt", name
        if "display" not in schematic.info:
            continue
        package, resource = _info(schematic, "display")["symbol"].split(":")
        svg = files(package) / resource
        assert svg.is_file(), name
        referenced.add(resource)

        match = re.search(r'viewBox="([^"]+)"', svg.read_text())
        assert match, f"{resource} has no viewBox"
        view_box = match.group(1)
        _, _, width, height = map(float, re.split(r"[\s,]+", view_box))
        box_width, box_height = _mosaic_box(_info(schematic, "ports"))
        assert width / height == pytest.approx(box_width / box_height), name

    packaged = {
        p.name for p in files("qpdk.symbols").iterdir() if p.name.endswith(".svg")
    }
    assert referenced == packaged


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
        double_pad_transmon: ("qpdk.models.qubit", {"left_pad", "right_pad"}),
        double_pad_transmon_with_bbox: (
            "qpdk.models.qubit",
            {"left_pad", "right_pad"},
        ),
        fluxonium: ("qpdk.models.qubit", {"left_pad", "right_pad"}),
        fluxonium_with_bbox: ("qpdk.models.qubit", {"left_pad", "right_pad"}),
        josephson_junction: (
            "qpdk.models.junction",
            {"left_wide", "right_wide"},
        ),
        unimon_coupled: ("qpdk.models.unimon", {"coupling_o3"}),
    }

    for cell, (module, ports) in expected.items():
        schematic = _get_schematic(cell)
        assert schematic is not None

        assert set(schematic.ports) == ports
        assert _info(schematic, "models")[0]["module"] == module
        assert set(_info(schematic, "models")[0]["port_order"]) == ports

    bend_s_schematic = _get_schematic(bend_s)
    assert bend_s_schematic is not None
    bend_s_model = _info(bend_s_schematic, "models")[0]
    assert bend_s_model["params"] == {}


def test_quarter_wave_resonator_coupled_symbol_port_sides() -> None:
    """Put the resonator end bottom-left, below the coupling input."""
    schematic = _get_schematic(quarter_wave_resonator_coupled)
    assert isinstance(schematic, DSchematic)

    assert [(p["name"], p["side"]) for p in _info(schematic, "ports")] == [
        ("coupling_o1", "left"),
        ("resonator_o1", "left"),
        ("coupling_o2", "right"),
    ]
    coupling_o1 = schematic.ports["coupling_o1"]
    resonator_o1 = schematic.ports["resonator_o1"]
    assert isinstance(coupling_o1, Port)
    assert isinstance(resonator_o1, Port)
    assert isinstance(coupling_o1.y, float)
    assert isinstance(resonator_o1.y, float)
    assert coupling_o1.y > resonator_o1.y
    assert resonator_o1.orientation == 180


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

        for descriptor in _info(schematic, "models"):
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


@pytest.mark.parametrize(
    "cell_name",
    sorted(PDK.cells.keys() & PDK.models.keys()),
    ids=lambda name: name,
)
def test_registered_models_expose_their_layout_ports(cell_name: str) -> None:
    """Every registered model simulates the ports its layout cell exposes."""
    layout_ports = {
        port.name
        for port in PDK.cells[cell_name]().ports
        if port.port_type != "placement"
    }
    s_params = PDK.models[cell_name](f=5e9)
    model_ports = {port for pair in s_params for port in pair}

    assert model_ports == layout_ports


def test_pdk_registration_follows_model_declarations() -> None:
    """Every registered layout model has a schematic declaring its boundary."""
    for name in PDK.cells.keys() & PDK.models.keys():
        schematic = _get_schematic(PDK.cells[name])
        assert schematic is not None, name
        descriptors = _info(schematic, "models")
        assert any(descriptor["language"] == "sax" for descriptor in descriptors), name


def test_analytical_models_without_a_layout_remain_registered() -> None:
    """Generic circuit primitives remain available without a layout declaration."""
    for name in sax_models.keys() - PDK.cells.keys():
        assert PDK.models[name] is sax_models[name]


def test_interdigital_capacitor_keeps_its_model_boundary() -> None:
    """The default two-plate layout matches the analytical model's ports.

    The single-sided variant is its own factory, ``interdigital_capacitor_half``,
    so the two-port model keeps one unambiguous port contract and stays
    registered for the sample chips that use it.
    """
    full = interdigital_capacitor()
    half = interdigital_capacitor_half()

    assert {port.name for port in full.ports} == {"o1", "o2"}
    assert {port.name for port in half.ports} == {"o1"}
    assert "interdigital_capacitor" in PDK.models
    assert "interdigital_capacitor_half" not in PDK.models


@pytest.mark.parametrize(("alias", "model_name"), SAX_MODEL_ALIASES.items())
def test_sax_model_aliases(alias: str, model_name: str) -> None:
    """Resolve aliased cells to the intended registered model."""
    assert PDK.models is not None
    assert model_name in sax_models
    assert alias in PDK.cells
    assert PDK.models[alias] is PDK.models[model_name]


def test_coupled_resonators_declare_model_boundaries() -> None:
    """Stop hierarchy expansion where capacitive coupling is modeled."""
    for cell in (resonator_coupled, quarter_wave_resonator_coupled):
        schematic = _get_schematic(cell)
        assert schematic is not None

        assert _info(schematic, "models")[0]["name"] == cell.__name__


def test_test_chip_does_not_declare_a_whole_chip_model() -> None:
    """Allow chip assemblies to expand until modeled component boundaries."""
    cell = PDK.cells["resonator_test_chip_python"]
    schematic = _get_schematic(cell)
    assert schematic is not None

    assert _info(schematic, "models") == []


@pytest.mark.parametrize(
    ("west", "east", "north", "south"), [(2, 1, 0, 1), (0, 2, 1, 0)]
)
def test_nxn_schematic_follows_the_layout_port_count(
    west: int, east: int, north: int, south: int
) -> None:
    """Junction schematics follow the requested number and sides of the ports."""
    settings = dict(west=west, east=east, north=north, south=south)
    layout = PDK.cells["nxn"](**settings)
    schematic = gf.kcl.factories["nxn"].get_schematic(**settings)

    assert set(schematic.ports) == {port.name for port in layout.ports}
    for name, port in schematic.ports.items():
        assert isinstance(port, Port)
        assert port.orientation == pytest.approx(layout.ports[name].orientation)
    assert _info(schematic, "models")[0]["port_order"] == list(schematic.ports)
