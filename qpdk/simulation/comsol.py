"""Build a minimal COMSOL 3D geometry project from a QPDK layout, via MPh.

This is the geometry milestone only. It imports the M1 metal polygons extracted
by :mod:`qpdk.simulation.comsol_layout` (holes preserved) into a COMSOL work
plane and extrudes them to a metal thickness. No RF physics, materials, ports,
or studies are added: an unsolved geometry project is not an RF simulation.

The MPh model is returned so a user can add the physics and study that fit
their problem, e.g. ``model.java.physics().create(...)``. Feed port coordinates
stay on the :class:`~qpdk.simulation.comsol_layout.ComsolLayout` that was passed
in; this module never claims a port was assigned in physics.

Example:
    >>> import mph
    >>> from qpdk.simulation.comsol import build_comsol_cpw_model
    >>> layout = prepare_comsol_layout(component)
    >>> model = build_comsol_cpw_model(client, layout, metal_thickness_um=0.2)
    >>> model.save("cpw.mph")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol_layout import ComsolLayout, Point


def _format_number(value: float) -> str:
    """Format a length for the COMSOL Java API.

    Returns:
        The value as a plain decimal string.
    """
    return f"{value:.12g}"


def _add_polygon(work_plane: Any, tag: str, points: tuple[Point, ...]) -> None:
    """Add one solid polygon to a work plane geometry sequence.

    Args:
        work_plane: The work plane's geometry sequence (``feature.geom()``).
        tag: Feature tag for the polygon.
        points: Vertices in µm, first vertex not repeated.
    """
    work_plane.create(tag, "Polygon")
    polygon = work_plane.feature(tag)
    polygon.set("x", ",".join(_format_number(point[0]) for point in points))
    polygon.set("y", ",".join(_format_number(point[1]) for point in points))


def build_comsol_cpw_model(
    client: mph.Client,
    layout: ComsolLayout,
    *,
    metal_thickness_um: float = 0.2,
    name: str = "QPDK CPW",
) -> mph.Model:
    """Create a COMSOL 3D model holding the layout's metal as extruded polygons.

    Each polygon outline becomes a solid polygon on work plane ``wp1``; each
    hole becomes a second polygon subtracted by its own ``Difference`` feature.
    The finished work plane is extruded to ``metal_thickness_um`` by ``ext1``.

    Args:
        client: A connected :class:`mph.Client`.
        layout: Extracted metal polygons and feed ports in µm.
        metal_thickness_um: Extrusion height in µm, strictly positive.
        name: Name of the COMSOL model.

    Returns:
        The MPh model. Geometry only: no physics, materials, ports, or studies
        have been added, and the model has not been saved.

    Raises:
        ValueError: If ``metal_thickness_um`` is not positive and finite, or
            the layout has no polygons to extrude.
    """
    if not math.isfinite(metal_thickness_um) or metal_thickness_um <= 0.0:
        raise ValueError(
            "metal_thickness_um must be positive and finite, "
            f"got {metal_thickness_um!r}"
        )
    if not layout.polygons:
        raise ValueError("layout has no polygons to extrude")

    model = client.create(name)
    model.java.component().create("comp1")
    geometry = model.java.component("comp1").geom().create("geom1", 3)
    geometry.lengthUnit("um")
    geometry.create("wp1", "WorkPlane")
    work_plane = geometry.feature("wp1").geom()

    for index, polygon in enumerate(layout.polygons):
        # One Difference per hole: selection().set() takes a single feature name.
        target = f"pol{index}"
        _add_polygon(work_plane, target, polygon.outline)

        for hole, points in enumerate(polygon.holes):
            hole_tag = f"hole{index}_{hole}"
            _add_polygon(work_plane, hole_tag, points)

            difference_tag = f"dif{index}_{hole}"
            work_plane.create(difference_tag, "Difference")
            difference = work_plane.feature(difference_tag)
            difference.selection("input").set(target)
            difference.selection("input2").set(hole_tag)
            target = difference_tag

    geometry.create("ext1", "Extrude")
    extrude = geometry.feature("ext1")
    extrude.set("workplane", "wp1")
    extrude.selection("input").set("wp1")
    extrude.set("distance", _format_number(metal_thickness_um))
    geometry.run()

    return model
