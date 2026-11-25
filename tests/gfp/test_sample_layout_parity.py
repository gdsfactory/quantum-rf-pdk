"""Check that every sample schematic builds the layout its netlist declares.

Each checked-in ``.gsch``/``.pic.yml`` pair describes one chip through both
editors. The ``.pic.yml`` layout is canonical, so materializing the schematic
with the installed GDSFactory+ SDK must reproduce its external ports and its
geometry on every fabrication layer. Only non-fabrication layers may differ,
and then only by zero-area marker artifacts such as labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.tech import LAYER, NON_METADATA_LAYERS

converter = import_gfp_module("gdsfactoryplus.nyancir_to_dschematic")
materializer = import_gfp_module("gdsfactoryplus.dschematic_to_gds")

_SAMPLE_DIR = Path(__file__).parents[2] / "qpdk/samples"
_SAMPLE_STEMS = tuple(
    path.stem
    for path in sorted(_SAMPLE_DIR.glob("*.gsch"))
    if path.with_name(f"{path.stem}.pic.yml").is_file()
)
_POSITION_TOLERANCE_UM = 1e-6
_FABRICATION_LAYERS = {tuple(layer) for layer in NON_METADATA_LAYERS}


def _layer_tuples(cell: gf.Component) -> set[tuple[int, int]]:
    layout = cell.kcl.layout
    return {(info.layer, info.datatype) for info in layout.layer_infos()}


def _port_layer(cell: gf.Component, layer: int) -> tuple[int, int]:
    info = cell.kcl.layout.get_info(layer)
    return info.layer, info.datatype


@pytest.mark.gfp
def test_sample_pairs_are_discovered() -> None:
    """The parity check must not pass vacuously on an empty sample directory."""
    assert _SAMPLE_STEMS


@pytest.mark.gfp
@pytest.mark.parametrize("stem", _SAMPLE_STEMS)
def test_sample_layout_matches_pic_yml(stem: str) -> None:
    """Materialize a schematic and require an exact match with its layout."""
    PDK.activate()
    schematic = _SAMPLE_DIR / f"{stem}.gsch"
    converted = converter.nyancir_to_dschematic(str(schematic), "qpdk.PDK")
    cell, warnings = materializer.materialize_dschematic(
        converted.top,
        kcl=converted.kcl,
        schem_name=f"{stem}_parity",
        factories=converted.factories,
    )
    assert warnings == [], stem

    fallback = gf.read.from_yaml(
        _SAMPLE_DIR / f"{stem}.pic.yml", name=f"{stem}_fallback_parity"
    )
    assert cell.dbbox() == fallback.dbbox(), stem
    assert {port.name for port in cell.ports} == {
        port.name for port in fallback.ports
    }, stem
    for port in fallback.ports:
        actual = cell.ports[port.name]
        assert actual.center == pytest.approx(
            port.center, abs=_POSITION_TOLERANCE_UM, rel=0
        )
        assert actual.width == port.width, (stem, port.name)
        assert actual.orientation == port.orientation, (stem, port.name)
        assert actual.port_type == port.port_type, (stem, port.name)
        assert _port_layer(cell, actual.layer) == _port_layer(fallback, port.layer), (
            stem,
            port.name,
        )

    for key, route in json.loads(schematic.read_text()).items():
        if route["type"] != "polyline":
            continue
        assert "vertices" in route, (
            f"{stem}: route {key} has no native vertices, and Livewire ignores "
            "the legacy waypoints format"
        )
        vertices = route["vertices"]
        assert len(vertices) >= 2, (
            f"{stem}: route {key} needs its two endpoint vertices, got {vertices}"
        )
        for side, index in (("start", 0), ("finish", -1)):
            terminal = route["terminals"][side]
            instance = fallback.insts[terminal["component_id"]]
            port = instance.ports[terminal["port_name"]]
            assert vertices[index] == pytest.approx(
                (port.x, -port.y), abs=fallback.kcl.dbu, rel=0
            ), (stem, key, side)

    for layer in sorted(_layer_tuples(cell) | _layer_tuples(fallback)):
        actual = gf.kdb.Region(cell.begin_shapes_rec(cell.kcl.layout.layer(*layer)))
        expected = gf.kdb.Region(
            fallback.begin_shapes_rec(fallback.kcl.layout.layer(*layer))
        )
        difference = actual ^ expected
        # Label-only layers report a non-empty region with zero area.
        if difference.area() == 0 and layer not in _FABRICATION_LAYERS:
            continue
        assert difference.is_empty(), (
            f"{stem}: layer {layer} differs from {stem}.pic.yml "
            f"(XOR area {difference.area()} dbu²)"
        )

    error = gf.kdb.Region(
        cell.begin_shapes_rec(cell.kcl.layout.layer(*LAYER.ERROR_PATH))
    )
    assert error.is_empty(), f"{stem}: GF+ left error markers behind"
