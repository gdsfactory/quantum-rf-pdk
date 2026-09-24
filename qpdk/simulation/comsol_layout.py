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
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        check that against ``polygons`` and ``bbox`` itself.
    """

    polygons: tuple[ComsolPolygon, ...]
    feed_ports: tuple[ComsolFeedPort, ...]
    bbox: ComsolBoundingBox


def prepare_comsol_layout(
    component: Component,
    feed_ports: tuple[str, str] | None = ("coupling_o1", "coupling_o2"),
    ground_margin: float = 100.0,
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

    Returns:
        A :class:`ComsolLayout` with metal polygons (holes preserved), the
        selected feed ports (empty when ``feed_ports`` is ``None``), and the
        prepared bounding box, all in µm.

    Raises:
        ValueError: If ``ground_margin`` is not positive and finite, if
            ``feed_ports`` is supplied but does not name two distinct available
            ports, if a feed port is not cardinal or has non-finite coordinates
            or width, if the component has no M1_ETCH mask or carries geometry
            on unsupported fabrication layers, or if no M1_DRAW metal remains.
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

    _reject_unsupported_layers(component)
    if not any(component.get_polygons(by="tuple", layers=[LAYER.M1_ETCH]).values()):
        raise ValueError(
            "COMSOL layout extraction requires an M1_ETCH mask to define the "
            "gaps; M1_DRAW shapes alone cannot be inverted into a ground plane"
        )

    prepared = prepare_metal_layout(
        component,
        margin_draw=ground_margin,
        name=f"{component.name}_comsol_{uuid.uuid4().hex}",
    )

    polygons = _extract_metal_polygons(prepared)
    if not polygons:
        raise ValueError(
            f"Component {component.name!r} has no M1_DRAW metal polygons to extract"
        )

    bbox = prepared.bbox()
    return ComsolLayout(
        polygons=polygons,
        feed_ports=ports,
        bbox=ComsolBoundingBox(
            xmin=bbox.left,
            ymin=bbox.bottom,
            xmax=bbox.right,
            ymax=bbox.top,
        ),
    )


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


def _extract_metal_polygons(component: Component) -> tuple[ComsolPolygon, ...]:
    """Return M1_DRAW polygons with their holes, in µm.

    Returns:
        The metal polygons, holes included.
    """
    dbu = component.kcl.dbu
    polygons = [
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
        for shapes in component.get_polygons(
            merge=True, by="tuple", layers=[LAYER.M1_DRAW]
        ).values()
        for shape in shapes
    ]
    return tuple(polygons)


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
