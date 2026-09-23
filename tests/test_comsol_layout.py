"""Tests for the COMSOL layout extraction.

These check the pure geometry and validation in
:mod:`qpdk.simulation.comsol_layout` on small QPDK CPW and unfed shapes. No
COMSOL or MPh involvement.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from qpdk.cells.resonator import quarter_wave_resonator_coupled
from qpdk.simulation.aedt_base import prepare_component_for_aedt
from qpdk.simulation.comsol_layout import ComsolLayout, prepare_comsol_layout
from qpdk.simulation.layout import prepare_metal_layout
from qpdk.tech import LAYER


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Return the area of a polygon given as an open vertex list (µm²).

    Returns:
        The polygon area in µm².
    """
    return (
        abs(
            sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
            )
        )
        / 2
    )


def _metal_area(layout: ComsolLayout) -> float:
    """Net metal area with holes subtracted (µm²).

    Returns:
        The total metal area in µm².
    """
    return sum(
        _polygon_area(polygon.outline)
        - sum(_polygon_area(hole) for hole in polygon.holes)
        for polygon in layout.polygons
    )


def _geometry_signature(component: gf.Component) -> dict:
    """Canonical per-layer polygon outlines and holes of a component.

    Returns:
        Layer specs mapped to sorted polygon outlines and holes.
    """
    signature = {}
    for layer, shapes in component.get_polygons(by="tuple", merge=True).items():
        polys = []
        for shape in shapes:
            outline = tuple((point.x, point.y) for point in shape.each_point_hull())
            holes = tuple(
                tuple((point.x, point.y) for point in shape.each_point_hole(hole))
                for hole in range(shape.holes())
            )
            polys.append((outline, holes))
        signature[layer] = sorted(polys)
    return signature


def test_aedt_wrapper_matches_shared_helper():
    """The AEDT entry point is geometrically identical to the shared helper."""
    comp = gf.components.straight(length=200, cross_section="cpw")

    via_wrapper = prepare_component_for_aedt(
        comp, margin_draw=40, margin_etch=5, name="equiv_aedt"
    )
    via_helper = prepare_metal_layout(
        comp, margin_draw=40, margin_etch=5, name="equiv_metal"
    )

    assert _geometry_signature(via_wrapper) == _geometry_signature(via_helper)
    wrapper_bbox, helper_bbox = via_wrapper.bbox(), via_helper.bbox()
    assert (
        wrapper_bbox.left,
        wrapper_bbox.bottom,
        wrapper_bbox.right,
        wrapper_bbox.top,
    ) == (
        helper_bbox.left,
        helper_bbox.bottom,
        helper_bbox.right,
        helper_bbox.top,
    )
    assert {p.name for p in via_wrapper.ports} == {p.name for p in via_helper.ports}


def test_straight_cpw_metal_area_and_ports():
    """Metal fills the ground box minus the CPW gaps and keeps the feed ports."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    margin = 50.0
    layout = prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=margin)

    # The prepared ground box is the component bbox grown by the margin on all sides.
    source = comp.bbox()
    assert layout.bbox.xmin == pytest.approx(source.left - margin)
    assert layout.bbox.ymin == pytest.approx(source.bottom - margin)
    assert layout.bbox.xmax == pytest.approx(source.right + margin)
    assert layout.bbox.ymax == pytest.approx(source.top + margin)

    # Default CPW cross-section: width=10, gap=6. The two gap strips of length
    # 200 must be voids, not metal.
    etch_area = 2 * 6.0 * 200.0
    assert _metal_area(layout) == pytest.approx(
        layout.bbox.width * layout.bbox.height - etch_area
    )
    # Negative space is preserved as holes rather than swallowed by the outline.
    assert any(polygon.holes for polygon in layout.polygons)

    ports = {port.name: port for port in layout.feed_ports}
    assert set(ports) == {"o1", "o2"}
    for name in ("o1", "o2"):
        source_port = comp.ports[name]
        assert ports[name].center == pytest.approx(tuple(source_port.center))
        assert ports[name].width == pytest.approx(source_port.width)
        assert ports[name].orientation == pytest.approx(source_port.orientation)


def test_coupled_resonator_feed_ports_and_holes():
    """A small QPDK coupled resonator extracts its coupling feeds and metal."""
    comp = quarter_wave_resonator_coupled(length=1000, meanders=2)
    margin = 50.0
    layout = prepare_comsol_layout(comp, ground_margin=margin)

    assert {port.name for port in layout.feed_ports} == {
        "coupling_o1",
        "coupling_o2",
    }
    for name in ("coupling_o1", "coupling_o2"):
        source_port = comp.ports[name]
        port = next(p for p in layout.feed_ports if p.name == name)
        assert port.center == pytest.approx(tuple(source_port.center))
        assert port.width == pytest.approx(source_port.width)

    # The ground plane surrounds the CPW, so the gap regions are holes.
    assert any(polygon.holes for polygon in layout.polygons)
    assert _metal_area(layout) > 0

    source = comp.bbox()
    assert layout.bbox.xmin == pytest.approx(source.left - margin)
    assert layout.bbox.ymin == pytest.approx(source.bottom - margin)
    assert layout.bbox.xmax == pytest.approx(source.right + margin)
    assert layout.bbox.ymax == pytest.approx(source.top + margin)


def test_no_feed_extraction_keeps_metal_and_bbox():
    """An unfed layout extracts the same metal but carries no feed ports."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    margin = 50.0
    layout = prepare_comsol_layout(comp, feed_ports=None, ground_margin=margin)

    assert layout.feed_ports == ()

    source = comp.bbox()
    assert layout.bbox.xmin == pytest.approx(source.left - margin)
    assert layout.bbox.ymin == pytest.approx(source.bottom - margin)
    assert layout.bbox.xmax == pytest.approx(source.right + margin)
    assert layout.bbox.ymax == pytest.approx(source.top + margin)

    fed = prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=margin)
    assert layout.polygons == fed.polygons
    assert _metal_area(layout) == pytest.approx(_metal_area(fed))


def test_no_feed_extraction_still_rejects_junction_layers():
    """JJ layers are refused, not skipped, so the caller must strip them itself."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=LAYER.JJ_AREA)

    with pytest.raises(ValueError, match="JJ_AREA"):
        prepare_comsol_layout(comp, feed_ports=None, ground_margin=50.0)


def test_reextracting_same_component_with_new_margin():
    """Notebook reruns can change the margin without a duplicate-cell error."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    first = prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=50)
    second = prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=75)

    assert second.bbox.width == pytest.approx(first.bbox.width + 50)
    assert second.bbox.height == pytest.approx(first.bbox.height + 50)
    assert second.feed_ports == first.feed_ports


@pytest.mark.parametrize("margin", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_margin(margin: float):
    """Ground margin must be strictly positive and finite."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    with pytest.raises(ValueError, match="ground_margin"):
        prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=margin)


def test_rejects_unknown_feed_port():
    """A feed name that is not a component port is rejected."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    with pytest.raises(ValueError, match="missing"):
        prepare_comsol_layout(comp, feed_ports=("o1", "missing"), ground_margin=50.0)


def test_rejects_duplicate_feed_ports():
    """The two feeds must be distinct."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    with pytest.raises(ValueError, match="distinct"):
        prepare_comsol_layout(comp, feed_ports=("o1", "o1"), ground_margin=50.0)


def test_rejects_non_cardinal_feed_port():
    """A rotated feed port is not axis-aligned and must be rejected."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    ref.rotate(45)
    comp.add_ports(ref.ports)

    with pytest.raises(ValueError, match="cardinal"):
        prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=50.0)


def test_rejects_unsupported_fabrication_layer():
    """Geometry on a layer other than M1 is refused, not silently dropped."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M2_DRAW)

    with pytest.raises(ValueError, match="M2_DRAW"):
        prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=50.0)


def test_rejects_positive_metal_without_etch_mask():
    """A positive metal rectangle does not define the CPW gaps or ground."""
    comp = gf.Component()
    comp.add_polygon([(0, 0), (200, 0), (200, 10), (0, 10)], layer=LAYER.M1_DRAW)
    comp.add_port(
        name="o1", center=(0, 5), width=10, orientation=180, layer=LAYER.M1_DRAW
    )
    comp.add_port(
        name="o2", center=(200, 5), width=10, orientation=0, layer=LAYER.M1_DRAW
    )

    with pytest.raises(ValueError, match="requires an M1_ETCH mask"):
        prepare_comsol_layout(comp, feed_ports=("o1", "o2"))
