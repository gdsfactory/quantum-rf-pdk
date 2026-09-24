"""Tests for the COMSOL layout extraction.

These check the pure geometry and validation in
:mod:`qpdk.simulation.comsol_layout` on small QPDK CPW and unfed shapes. No
COMSOL or MPh involvement.
"""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb
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


def _kdb_polygon(points: tuple[tuple[float, float], ...], dbu: float) -> kdb.Polygon:
    """Rebuild a µm vertex list as a KLayout polygon in database units.

    Returns:
        The polygon in database units.
    """
    return kdb.Polygon([kdb.Point(round(x / dbu), round(y / dbu)) for x, y in points])


def _in_metal(layout: ComsolLayout, point: tuple[float, float], dbu: float) -> bool:
    """Return whether a µm point falls on extracted metal (holes excluded).

    Returns:
        ``True`` when the point lies inside some outline and no hole of it.
    """
    probe = kdb.Point(round(point[0] / dbu), round(point[1] / dbu))
    for polygon in layout.polygons:
        if not _kdb_polygon(polygon.outline, dbu).inside(probe):
            continue
        if any(_kdb_polygon(hole, dbu).inside(probe) for hole in polygon.holes):
            continue
        return True
    return False


def test_aedt_wrapper_matches_shared_helper():
    """The AEDT wrapper forwards both margins and ports to the shared helper."""
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


def test_etch_margin_widens_the_cpw_gap():
    """A positive etch margin removes ground metal beside the CPW gap."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    plain = prepare_metal_layout(comp, margin_draw=50, name="plain_etch_margin")
    widened = prepare_metal_layout(
        comp, margin_draw=50, margin_etch=2, name="widened_etch_margin"
    )
    point = kdb.Point(round(100 / plain.kcl.dbu), round(12 / plain.kcl.dbu))

    def contains(component: gf.Component) -> bool:
        shapes = component.get_polygons(merge=True, by="tuple", layers=[LAYER.M1_DRAW])
        return any(shape.inside(point) for shape in shapes[tuple(LAYER.M1_DRAW)])

    assert contains(plain)
    assert not contains(widened)


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


@pytest.mark.parametrize("orientation", [89.9999999, 179.9999999])
def test_accepts_roundoff_below_cardinal_angle(orientation: float):
    """Roundoff on either side of a cardinal angle is accepted."""
    comp = gf.Component()
    comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_port(
        name="near_cardinal",
        center=(0, 0),
        width=10,
        orientation=orientation,
        layer=LAYER.M1_DRAW,
    )
    comp.add_port(
        name="other",
        center=(200, 0),
        width=10,
        orientation=0,
        layer=LAYER.M1_DRAW,
    )

    layout = prepare_comsol_layout(
        comp, feed_ports=("near_cardinal", "other"), ground_margin=50.0
    )
    assert layout.feed_ports[0].orientation == pytest.approx(orientation)


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


def test_crop_to_feed_ports_defaults_to_off():
    """Cropping is opt-in; the default keeps the prepared ground untouched."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    margin = 50.0
    default = prepare_comsol_layout(comp, feed_ports=("o1", "o2"), ground_margin=margin)
    explicit = prepare_comsol_layout(
        comp, feed_ports=("o1", "o2"), ground_margin=margin, crop_to_feed_ports=False
    )

    assert default.polygons == explicit.polygons
    assert default.bbox == explicit.bbox
    source = comp.bbox()
    assert default.bbox.xmin == pytest.approx(source.left - margin)
    assert default.bbox.xmax == pytest.approx(source.right + margin)


def test_crop_to_feed_ports_opens_both_x_faces():
    """Cropping exposes conductor/gap/ground separation at both feed planes."""
    comp = gf.components.straight(length=200, cross_section="cpw")
    margin = 50.0
    dbu = comp.kcl.dbu
    layout = prepare_comsol_layout(
        comp, feed_ports=("o1", "o2"), ground_margin=margin, crop_to_feed_ports=True
    )

    source = comp.bbox()
    assert layout.bbox.xmin == pytest.approx(source.left)
    assert layout.bbox.xmax == pytest.approx(source.right)
    assert layout.bbox.ymin == pytest.approx(source.bottom - margin)
    assert layout.bbox.ymax == pytest.approx(source.top + margin)

    # Probe just inside each feed plane: the CPW cross-section is open, i.e.
    # the center conductor is present and both etch gaps and the ground are
    # separated from it.
    for x in (0.5, 199.5):
        assert _in_metal(layout, (x, 0.0), dbu)
        assert not _in_metal(layout, (x, 8.0), dbu)
        assert not _in_metal(layout, (x, -8.0), dbu)
        assert _in_metal(layout, (x, 40.0), dbu)
        assert _in_metal(layout, (x, -40.0), dbu)
    # The ground no longer extends past the feed planes.
    assert not _in_metal(layout, (-1.0, 40.0), dbu)
    assert not _in_metal(layout, (201.0, 40.0), dbu)


def test_crop_to_feed_ports_supports_vertical_feeds():
    """The y axis works the same way: bottom 270°, top 90°."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    ref.rotate(90)
    comp.add_ports(ref.ports)
    margin = 50.0
    dbu = comp.kcl.dbu
    layout = prepare_comsol_layout(
        comp, feed_ports=("o1", "o2"), ground_margin=margin, crop_to_feed_ports=True
    )

    source = comp.bbox()
    assert layout.bbox.ymin == pytest.approx(source.bottom)
    assert layout.bbox.ymax == pytest.approx(source.top)
    assert layout.bbox.xmin == pytest.approx(source.left - margin)
    assert layout.bbox.xmax == pytest.approx(source.right + margin)

    for y in (0.5, 199.5):
        assert _in_metal(layout, (0.0, y), dbu)
        assert not _in_metal(layout, (8.0, y), dbu)
        assert not _in_metal(layout, (-8.0, y), dbu)
        assert _in_metal(layout, (40.0, y), dbu)


def test_crop_to_feed_ports_preserves_holes():
    """An enclosed void survives cropping as a hole in the ground metal."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_polygon([(90, 20), (110, 20), (110, 30), (90, 30)], layer=LAYER.M1_ETCH)

    layout = prepare_comsol_layout(
        comp, feed_ports=("o1", "o2"), ground_margin=50.0, crop_to_feed_ports=True
    )

    holes = [hole for polygon in layout.polygons for hole in polygon.holes]
    assert len(holes) == 1
    xs = [x for x, _ in holes[0]]
    ys = [y for _, y in holes[0]]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((
        90.0,
        20.0,
        110.0,
        30.0,
    ))


def test_crop_to_feed_ports_rejects_metal_beyond_the_planes():
    """A resonator wider than its coupling feeds must not be sliced."""
    comp = quarter_wave_resonator_coupled(length=1000, meanders=2)

    with pytest.raises(ValueError, match="cut component geometry"):
        prepare_comsol_layout(comp, ground_margin=50.0, crop_to_feed_ports=True)


def test_crop_to_feed_ports_rejects_etch_stopping_short():
    """Gaps that do not reach the planes would leave a shorted face."""
    comp = gf.Component()
    comp.add_polygon([(0, -5), (200, -5), (200, 5), (0, 5)], layer=LAYER.M1_DRAW)
    comp.add_polygon([(50, -11), (150, -11), (150, -5), (50, -5)], layer=LAYER.M1_ETCH)
    comp.add_polygon([(50, 5), (150, 5), (150, 11), (50, 11)], layer=LAYER.M1_ETCH)
    comp.add_port(
        name="o1", center=(0, 0), width=10, orientation=180, layer=LAYER.M1_DRAW
    )
    comp.add_port(
        name="o2", center=(200, 0), width=10, orientation=0, layer=LAYER.M1_DRAW
    )

    with pytest.raises(ValueError, match="etch gaps to reach both"):
        prepare_comsol_layout(
            comp, feed_ports=("o1", "o2"), ground_margin=50.0, crop_to_feed_ports=True
        )


def test_crop_to_feed_ports_requires_feeds():
    """Cropping has no planes to use when the layout is unfed."""
    comp = gf.components.straight(length=200, cross_section="cpw")

    with pytest.raises(ValueError, match="needs two feed ports"):
        prepare_comsol_layout(
            comp, feed_ports=None, ground_margin=50.0, crop_to_feed_ports=True
        )


def test_crop_to_feed_ports_rejects_inward_feeds():
    """Both feeds facing the same way is not a valid opposite pair."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_port(
        name="inward", center=(0, 0), width=10, orientation=0, layer=LAYER.M1_DRAW
    )
    comp.add_port(
        name="outward", center=(200, 0), width=10, orientation=0, layer=LAYER.M1_DRAW
    )

    with pytest.raises(ValueError, match="face outwards"):
        prepare_comsol_layout(
            comp,
            feed_ports=("inward", "outward"),
            ground_margin=50.0,
            crop_to_feed_ports=True,
        )


def test_crop_to_feed_ports_rejects_mixed_axes():
    """Feeds on different axes do not define a single crop plane pair."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_port(
        name="o1", center=(0, 0), width=10, orientation=180, layer=LAYER.M1_DRAW
    )
    comp.add_port(
        name="vertical", center=(200, 0), width=10, orientation=90, layer=LAYER.M1_DRAW
    )

    with pytest.raises(ValueError, match="same axis"):
        prepare_comsol_layout(
            comp,
            feed_ports=("o1", "vertical"),
            ground_margin=50.0,
            crop_to_feed_ports=True,
        )


def test_crop_to_feed_ports_rejects_misaligned_feeds():
    """Feeds that do not share a transverse coordinate cannot bound a strip."""
    comp = gf.Component()
    ref = comp << gf.components.straight(length=200, cross_section="cpw")
    comp.add_ports(ref.ports)
    comp.add_port(
        name="o1", center=(0, 0), width=10, orientation=180, layer=LAYER.M1_DRAW
    )
    comp.add_port(
        name="raised", center=(200, 5), width=10, orientation=0, layer=LAYER.M1_DRAW
    )

    with pytest.raises(ValueError, match="transverse coordinate"):
        prepare_comsol_layout(
            comp,
            feed_ports=("o1", "raised"),
            ground_margin=50.0,
            crop_to_feed_ports=True,
        )
