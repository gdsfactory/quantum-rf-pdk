"""Extract a gdsfactory component into COMSOL-ready metal polygons and ports.

This module is pure geometry: it converts a QPDK M1_ETCH mask into physical
M1_DRAW metal polygons (in micrometres) plus optional feed-port metadata so a
separate COMSOL model builder can consume it. Layouts that are not fed (e.g. an
eigenmode qubit cell) are extracted with ``feed_ports=None`` and carry no feed
metadata. No COMSOL, MPh, or PyAEDT imports live here.

The negative-mask trick lives in the neutral
:func:`~qpdk.simulation.layout.prepare_metal_layout`: it folds the additive
metal into the etch layer, then inverts the mask around the component bounding
box and applies the ground margin. The result exchanged here has positive
M1_DRAW metal, no etch layers, and the original ports re-added.

Optionally the prepared metal can be cropped at the two feed-port planes so a
component whose CPW conductors and both etch-gap strips reach those planes ends
in open CPW cross sections instead of solid ground.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import klayout.db as kdb

from qpdk.simulation.layout import prepare_metal_layout
from qpdk.tech import LAYER, NON_METADATA_LAYERS

if TYPE_CHECKING:
    from gdsfactory.component import Component

type Point = tuple[float, float]

#: Input layers that the extraction understands. Everything else in
#: :data:`~qpdk.tech.NON_METADATA_LAYERS` is rejected rather than dropped.
_SUPPORTED_LAYERS: frozenset[tuple[int, int]] = frozenset({
    tuple(LAYER.M1_DRAW),
    tuple(LAYER.M1_ETCH),
})

# Feed planes are currently defined only for axis-aligned ports.
_ANGLE_TOLERANCE = 1e-6

#: Coordinate tolerance in µm for feed-plane alignment and ordering.
_PLANE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True, kw_only=True)
class ComsolPolygon:
    """One physical metal polygon in µm.

    ``outline`` is the outer boundary (vertices in order, first vertex not
    repeated). ``holes`` are inner boundaries, i.e. etched negative space
    enclosed by metal.

    A single simple COMSOL polygon point list cannot encode a hole, so a
    consumer that only accepts plain outlines must either subtract ``holes``
    itself or reject polygons where ``holes`` is non-empty. ``outline`` alone
    is the complete shape only when ``holes`` is empty.
    """

    outline: tuple[Point, ...]
    holes: tuple[tuple[Point, ...], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ComsolFeedPort:
    """A feed port selected for the COMSOL model, in µm and degrees."""

    name: str
    center: Point
    width: float
    orientation: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ComsolBoundingBox:
    """Axis-aligned bounding box of the prepared ground region, in µm."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        """Box width in µm."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Box height in µm."""
        return self.ymax - self.ymin


@dataclass(frozen=True, slots=True, kw_only=True)
class ComsolLayout:
    """Extracted COMSOL geometry and feed ports for one component.

    All coordinates are in micrometres in the component's own frame.
    ``feed_ports`` is empty for layouts extracted with ``feed_ports=None``.

    Note:
        The feed ports are reported as-is. Whether each one actually sits on
        the outer boundary of the prepared ground region is *not* asserted
        here: after the mask inversion the ground plane surrounds the feeds,
        so a port center can legitimately lie inside the bounding box. A model
        builder that needs the port to coincide with a ground boundary must
        check that against ``polygons`` and ``bbox`` itself. The exception is
        an extraction made with ``crop_to_feed_ports=True``, where the two feed
        planes bound the metal and each feed center lies on a face of ``bbox``.
    """

    polygons: tuple[ComsolPolygon, ...]
    feed_ports: tuple[ComsolFeedPort, ...]
    bbox: ComsolBoundingBox


@dataclass(frozen=True, slots=True, kw_only=True)
class _CropPlanes:
    """The two feed-port planes a cropped extraction is bounded by, in µm.

    ``axis`` is 0 to crop along x, 1 to crop along y. ``low``/``high`` are the
    port centers on that axis, ordered low to high.
    """

    axis: int
    low: float
    high: float


def prepare_comsol_layout(
    component: Component,
    feed_ports: tuple[str, str] | None = ("coupling_o1", "coupling_o2"),
    ground_margin: float = 100.0,
    *,
    crop_to_feed_ports: bool = False,
) -> ComsolLayout:
    """Extract M1_DRAW metal polygons and optional feed ports from a component.

    Args:
        component: The gdsfactory component to extract. It must contain an
            M1_ETCH mask; positive M1_DRAW shapes alone do not define the gaps.
            The original ports are preserved.
        feed_ports: Names of exactly two distinct ports to expose as feeds, or
            ``None`` for an unfed layout, in which case the result carries no
            feed ports. ``None`` fits layouts with no CPW feedline, e.g. an
            eigenmode qubit cell.
        ground_margin: Positive margin in µm added around the component
            bounding box to form the ground plane.
        crop_to_feed_ports: When ``True``, cut the prepared metal back to the
            two feed-port planes so each external face shows an open CPW cross
            section instead of the extended ground. Requires two feed ports
            that are axis-aligned, opposite, outward-facing, and share their
            transverse coordinate (left/right along x, or bottom/top along y).
            Cropping is refused unless every M1_DRAW and M1_ETCH polygon of the
            input component lies between the planes and the etch gaps reach
            both planes, so no resonator metal is cut and no face is shorted.
            The planes snap outwards to the database grid, so the returned
            ``bbox`` can exceed the port span by less than one database unit.
            Defaults to ``False``, which leaves the prepared ground untouched.

    Returns:
        A :class:`ComsolLayout` with metal polygons (holes preserved), the
        selected feed ports (empty when ``feed_ports`` is ``None``), and the
        prepared bounding box, all in µm.

    Raises:
        ValueError: If ``ground_margin`` is not positive and finite, if
            ``feed_ports`` is supplied but does not name two distinct available
            ports, if a feed port is not cardinal or has non-finite coordinates
            or width, if the component has no M1_ETCH mask or carries geometry
            on unsupported fabrication layers, if no M1_DRAW metal remains, or
            if ``crop_to_feed_ports`` is set but the feeds are not a valid
            opposite pair or the component does not fit between the planes.
    """
    if not math.isfinite(ground_margin) or ground_margin <= 0.0:
        raise ValueError(
            f"ground_margin must be positive and finite, got {ground_margin!r}"
        )

    ports: tuple[ComsolFeedPort, ...]
    if feed_ports is None:
        ports = ()
    else:
        if len(feed_ports) != 2:
            raise ValueError(
                "feed_ports must name exactly two ports or be None, "
                f"got {len(feed_ports)}"
            )
        if feed_ports[0] == feed_ports[1]:
            raise ValueError(
                f"feed_ports must be two distinct names, got {feed_ports!r}"
            )
        # Validate feeds before preparation: preparing registers a new cell and
        # we do not want invalid input to leave that side effect behind.
        ports = tuple(_feed_port(component, name) for name in feed_ports)

    planes: _CropPlanes | None = None
    if crop_to_feed_ports:
        if not ports:
            raise ValueError(
                "crop_to_feed_ports=True needs two feed ports to define the "
                "crop planes, but feed_ports is None"
            )
        planes = _crop_planes(ports)

    _reject_unsupported_layers(component)
    if not any(component.get_polygons(by="tuple", layers=[LAYER.M1_ETCH]).values()):
        raise ValueError(
            "COMSOL layout extraction requires an M1_ETCH mask to define the "
            "gaps; M1_DRAW shapes alone cannot be inverted into a ground plane"
        )

    dbu = component.kcl.dbu
    low_dbu = high_dbu = 0
    if planes is not None:
        # Snap outwards so a sub-grid port offset never clips validated metal.
        low_dbu = math.floor(planes.low / dbu)
        high_dbu = math.ceil(planes.high / dbu)
        _validate_crop_containment(component, planes, low_dbu, high_dbu)

    prepared = prepare_metal_layout(
        component,
        margin_draw=ground_margin,
        name=f"{component.name}_comsol_{uuid.uuid4().hex}",
    )

    if planes is None:
        polygons = _extract_metal_polygons(prepared)
        if not polygons:
            raise ValueError(
                f"Component {component.name!r} has no M1_DRAW metal polygons to extract"
            )
        source = prepared.bbox()
        bbox = ComsolBoundingBox(
            xmin=source.left,
            ymin=source.bottom,
            xmax=source.right,
            ymax=source.top,
        )
    else:
        polygons, cropped = _crop_metal_polygons(prepared, planes, low_dbu, high_dbu)
        if not polygons:
            raise ValueError(
                f"Component {component.name!r} has no M1_DRAW metal between the "
                "feed planes to extract"
            )
        bbox = ComsolBoundingBox(
            xmin=cropped.left * dbu,
            ymin=cropped.bottom * dbu,
            xmax=cropped.right * dbu,
            ymax=cropped.top * dbu,
        )

    return ComsolLayout(polygons=polygons, feed_ports=ports, bbox=bbox)


def _reject_unsupported_layers(component: Component) -> None:
    """Refuse components carrying fabrication layers this extractor ignores.

    Only M1_DRAW (and the M1_ETCH it is inverted from) is modelled. Layers
    such as M2_DRAW, airbridges, or the JJ_AREA/JJ_PATCH junction layers would
    be silently lost, making the extracted geometry wrong rather than failed,
    so they are rejected up front. A layout that wants an EM-only copy (e.g. a
    qubit cell with the junction removed) must strip those layers itself before
    extraction. Checked on the input because the AEDT preparation discards
    additive metal layers.

    Raises:
        ValueError: If any unsupported fabrication layer carries geometry.
    """
    unsupported_specs = {
        tuple(layer): str(layer)
        for layer in NON_METADATA_LAYERS
        if tuple(layer) not in _SUPPORTED_LAYERS
    }
    present = sorted(
        unsupported_specs[spec]
        for spec, shapes in component.get_polygons(by="tuple").items()
        if shapes and spec in unsupported_specs
    )
    if present:
        raise ValueError(
            "COMSOL layout extraction supports only the M1_DRAW layer, but the "
            f"component also has geometry on {present}. Remove those layers or "
            "extend the extraction before building a COMSOL model."
        )


def _polygons_from_shapes(
    shapes: Iterable[kdb.Polygon], dbu: float
) -> tuple[ComsolPolygon, ...]:
    """Convert KLayout polygons to µm :class:`ComsolPolygon` values.

    Returns:
        One :class:`ComsolPolygon` per shape, holes preserved.
    """
    return tuple(
        ComsolPolygon(
            outline=tuple(
                (point.x * dbu, point.y * dbu) for point in shape.each_point_hull()
            ),
            holes=tuple(
                tuple(
                    (point.x * dbu, point.y * dbu)
                    for point in shape.each_point_hole(hole)
                )
                for hole in range(shape.holes())
            ),
        )
        for shape in shapes
    )


def _extract_metal_polygons(component: Component) -> tuple[ComsolPolygon, ...]:
    """Return M1_DRAW polygons with their holes, in µm.

    Returns:
        The metal polygons, holes included.
    """
    shapes = component.get_polygons(merge=True, by="tuple", layers=[LAYER.M1_DRAW]).get(
        tuple(LAYER.M1_DRAW), []
    )
    return _polygons_from_shapes(shapes, component.kcl.dbu)


def _crop_metal_polygons(
    component: Component, planes: _CropPlanes, low_dbu: int, high_dbu: int
) -> tuple[tuple[ComsolPolygon, ...], kdb.Box]:
    """Clip prepared M1_DRAW metal to the strip between the feed planes.

    Returns:
        The cropped polygons (holes preserved) and their bounding box, the
        latter in the component's database units.
    """
    region = kdb.Region()
    shapes = component.get_polygons(merge=True, by="tuple", layers=[LAYER.M1_DRAW]).get(
        tuple(LAYER.M1_DRAW), []
    )
    if shapes:
        region.insert(shapes)
    if region.is_empty():
        return (), kdb.Box()
    outer = region.bbox()
    clip_box = (
        kdb.Box(low_dbu, outer.bottom, high_dbu, outer.top)
        if planes.axis == 0
        else kdb.Box(outer.left, low_dbu, outer.right, high_dbu)
    )
    clip = kdb.Region()
    clip.insert(clip_box)
    cropped = region & clip
    cropped.merge()
    return _polygons_from_shapes(cropped.each(), component.kcl.dbu), cropped.bbox()


def _axial_bounds(box: kdb.Box, axis: int) -> tuple[int, int]:
    """Return the low and high extent of a box along the crop axis.

    Returns:
        The box extents in database units along ``axis``.
    """
    return (box.left, box.right) if axis == 0 else (box.bottom, box.top)


def _axis_of_orientation(orientation: float) -> int:
    """Return the crop axis a cardinal port faces along, 0 for x and 1 for y.

    Returns:
        ``0`` when the port faces along ±x, ``1`` along ±y.
    """
    remainder = orientation % 180.0
    return 0 if min(remainder, 180.0 - remainder) <= _ANGLE_TOLERANCE else 1


def _same_angle(first: float, second: float) -> bool:
    """Return whether two orientations point the same way modulo 360°.

    Returns:
        ``True`` when the orientations differ by less than
        :data:`_ANGLE_TOLERANCE`.
    """
    return abs((first - second + 180.0) % 360.0 - 180.0) <= _ANGLE_TOLERANCE


def _crop_planes(ports: tuple[ComsolFeedPort, ...]) -> _CropPlanes:
    """Validate the feed pair and derive the two crop planes.

    Returns:
        The axis and ordered planes of the valid feed pair.

    Raises:
        ValueError: If the feeds are not two axis-aligned, opposite,
            outward-facing ports sharing a transverse coordinate.
    """
    axes = {_axis_of_orientation(port.orientation) for port in ports}
    if len(axes) != 1:
        raise ValueError(
            "crop_to_feed_ports requires both feed ports to face along the same "
            f"axis, got orientations {[port.orientation for port in ports]}"
        )
    axis = axes.pop()
    low_port, high_port = sorted(ports, key=lambda port: port.center[axis])
    expected = (180.0, 0.0) if axis == 0 else (270.0, 90.0)
    if not (
        _same_angle(low_port.orientation, expected[0])
        and _same_angle(high_port.orientation, expected[1])
    ):
        raise ValueError(
            "crop_to_feed_ports requires the feeds to face outwards, "
            f"{expected[0]} on the low side and {expected[1]} on the high side, "
            f"got {low_port.orientation} and {high_port.orientation}"
        )
    transverse = 1 - axis
    if (
        abs(low_port.center[transverse] - high_port.center[transverse])
        > _PLANE_TOLERANCE
    ):
        raise ValueError(
            "crop_to_feed_ports requires both feed centers to share their "
            f"transverse coordinate, got {low_port.center} and {high_port.center}"
        )
    if high_port.center[axis] - low_port.center[axis] <= _PLANE_TOLERANCE:
        raise ValueError(
            "crop_to_feed_ports requires two distinct ordered feed planes, got "
            f"{low_port.center[axis]} and {high_port.center[axis]}"
        )
    return _CropPlanes(
        axis=axis, low=low_port.center[axis], high=high_port.center[axis]
    )


def _validate_crop_containment(
    component: Component, planes: _CropPlanes, low_dbu: int, high_dbu: int
) -> None:
    """Refuse a crop that would cut component geometry or leave a closed face.

    Every M1_DRAW and M1_ETCH polygon of the input must lie between the planes,
    otherwise cropping would slice resonator metal. The etch must also reach
    both planes, otherwise the cropped faces would be solid metal rather than
    open CPW cross sections.

    Raises:
        ValueError: If component geometry extends beyond a plane or the etch
            does not span both planes.
    """
    for spec in (LAYER.M1_DRAW, LAYER.M1_ETCH):
        for shape in component.get_polygons(by="tuple", layers=[spec]).get(
            tuple(spec), []
        ):
            low, high = _axial_bounds(shape.bbox(), planes.axis)
            if low < low_dbu or high > high_dbu:
                raise ValueError(
                    "crop_to_feed_ports would cut component geometry: "
                    f"{spec} extends beyond the feed planes at "
                    f"[{planes.low}, {planes.high}] µm. Move the feeds to the "
                    "component ends or drop cropping."
                )

    etch_shapes = component.get_polygons(by="tuple", layers=[LAYER.M1_ETCH]).get(
        tuple(LAYER.M1_ETCH), []
    )
    if not etch_shapes:
        raise ValueError("crop_to_feed_ports requires etch gaps at both feed planes")
    etch = kdb.Region()
    etch.insert(etch_shapes)
    low, high = _axial_bounds(etch.bbox(), planes.axis)
    if low > low_dbu or high < high_dbu:
        raise ValueError(
            "crop_to_feed_ports requires the etch gaps to reach both feed "
            f"planes at [{planes.low}, {planes.high}] µm, otherwise the cropped "
            "faces are not open CPW cross sections"
        )


def _feed_port(component: Component, name: str) -> ComsolFeedPort:
    """Validate one feed port and return its metadata, in µm and degrees.

    Returns:
        The validated feed port.

    Raises:
        ValueError: If the port is unavailable, non-cardinal, or has
            non-finite or non-positive values.
    """
    try:
        port = component.ports[name]
    except KeyError as exc:
        available = sorted(port_.name for port_ in component.ports)
        raise ValueError(
            f"Feed port {name!r} is not available; component has {available}"
        ) from exc

    center: Point = (float(port.center[0]), float(port.center[1]))
    width = float(port.width)
    orientation = float(port.orientation)

    for label, value in (
        ("x coordinate", center[0]),
        ("y coordinate", center[1]),
        ("width", width),
        ("orientation", orientation),
    ):
        if not math.isfinite(value):
            raise ValueError(f"Feed port {name!r} has non-finite {label}")
    if width <= 0.0:
        raise ValueError(f"Feed port {name!r} has non-positive width {width}")
    remainder = orientation % 90.0
    if min(remainder, 90.0 - remainder) > _ANGLE_TOLERANCE:
        raise ValueError(
            f"Feed port {name!r} orientation {orientation} is not cardinal; "
            "COMSOL feeds must face along ±x or ±y"
        )

    return ComsolFeedPort(
        name=name, center=center, width=width, orientation=orientation
    )
