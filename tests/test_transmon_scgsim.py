"""Technical invariants for the opt-in transmon EM sheet and authored topology."""

import math

import gdsfactory as gf
import pytest
from shapely.geometry import Point, Polygon

from qpdk.cells.transmon import double_pad_transmon
from qpdk.tech import LAYER, NON_METADATA_LAYERS, get_layer_stack, material_properties


def test_lumped_sheet_is_opt_in_and_preserves_fabrication():
    plain = double_pad_transmon()
    simulated = double_pad_transmon(with_junction_lumped_port=True)
    plain_polygons = plain.get_polygons_points(by="tuple")
    sim_polygons = simulated.get_polygons_points(by="tuple")
    boundary = gf.get_layer_tuple(LAYER.SIM_BOUNDARY)
    assert boundary not in plain_polygons
    assert "junction_lumped" not in [port.name for port in plain.ports]
    assert set(sim_polygons) - set(plain_polygons) == {boundary}
    for layer, polygons in plain_polygons.items():
        assert len(polygons) == len(sim_polygons[layer])
        for original, candidate in zip(polygons, sim_polygons[layer], strict=True):
            assert Polygon(original).equals(Polygon(candidate))
    for port in plain.ports:
        candidate = simulated.ports[port.name]
        assert candidate.center == port.center
        assert candidate.width == port.width
        assert candidate.orientation == port.orientation
        assert candidate.layer == port.layer
    assert LAYER.SIM_BOUNDARY not in NON_METADATA_LAYERS


@pytest.mark.parametrize(
    ("pad_size", "pad_gap", "width"), [((250, 400), 15, 1), ((80, 120), 9, 2)]
)
def test_sheet_contacts_both_pads(pad_size, pad_gap, width):
    component = double_pad_transmon(
        pad_size=pad_size,
        pad_gap=pad_gap,
        with_junction_lumped_port=True,
        junction_lumped_port_width=width,
    )
    polygons = component.get_polygons_points(by="tuple")
    sheets = polygons[gf.get_layer_tuple(LAYER.SIM_BOUNDARY)]
    assert len(sheets) == 1
    sheet = Polygon(sheets[0])
    pads = [Polygon(p) for p in polygons[gf.get_layer_tuple(LAYER.M1_DRAW)]]
    assert len(pads) == 2
    assert all(sheet.intersection(pad).area > 0 for pad in pads)
    assert sheet.bounds[0] == pytest.approx(-pad_gap / 2 - component.kcl.dbu)
    assert sheet.bounds[2] == pytest.approx(pad_gap / 2 + component.kcl.dbu)
    assert sheet.bounds[3] - sheet.bounds[1] == pytest.approx(width)
    port = component.ports["junction_lumped"]
    assert port.center == (0, 0)
    assert port.width == width
    assert port.orientation == 0
    assert port.layer == gf.get_layer(LAYER.SIM_BOUNDARY)


@pytest.mark.parametrize("width", [0, -1, math.inf, math.nan, 401])
def test_invalid_enabled_sheet_width(width):
    with pytest.raises(ValueError, match="positive gap and width"):
        double_pad_transmon(
            with_junction_lumped_port=True, junction_lumped_port_width=width
        )


@pytest.mark.parametrize(
    "layer", sorted(NON_METADATA_LAYERS | {LAYER.SIM_AREA}, key=int)
)
def test_sheet_cannot_share_metal_layer(layer):
    with pytest.raises(ValueError, match="non-fabrication layer"):
        double_pad_transmon(with_junction_lumped_port=True, layer_simulation=layer)


@pytest.mark.parametrize(("pad_size", "pad_gap"), [((250, 400), 15), ((80, 120), 9)])
def test_semantics_select_actual_pad_centers(pad_size, pad_gap):
    component = double_pad_transmon(pad_size=pad_size, pad_gap=pad_gap)
    semantics = component.info["component_semantics"]
    assert semantics["schema_version"] == 1
    regions = semantics["conductor_regions"]
    assert [(r["semantic_id"], r["net_id"]) for r in regions] == [
        ("LEFT_PAD", "left_pad"),
        ("RIGHT_PAD", "right_pad"),
    ]
    pads = [
        Polygon(p)
        for p in component.get_polygons_points(by="tuple")[
            gf.get_layer_tuple(LAYER.M1_DRAW)
        ]
    ]
    for region, sign in zip(regions, [-1, 1], strict=True):
        assert region["level"] == "M1"
        assert region["gds_layer"] == list(gf.get_layer_tuple(LAYER.M1_DRAW))
        selector = region["geometry"]["selector_point_um"]
        assert selector == [sign * (pad_size[0] + pad_gap) / 2, 0]
        assert sum(pad.contains(Point(selector)) for pad in pads) == 1


def test_process_facts_retain_existing_dimensions():
    stack = get_layer_stack()
    metal = stack.layers["M1"]
    assert metal.thickness == pytest.approx(0.2)
    assert metal.zmin == 0
    assert metal.info["simulation_role"] == "conductor"
    assert metal.info["host_void_semantic_id"] == "Vacuum"
    assert metal.info["part_role"] == "face_metal"
    for name in ["Substrate", "Vacuum"]:
        assert stack.layers[name].info["simulation_role"] == "solution_region"
        assert stack.layers[name].info["include_in_component_simulation"] is True
    assert material_properties["Nb"]["material_kind"] == "superconductor"
    assert material_properties["Si"]["relative_permittivity"] == pytest.approx(11.45)
    assert material_properties["Si"]["material_kind"] == "dielectric"
    assert material_properties["vacuum"]["material_kind"] == "vacuum"
