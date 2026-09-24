r"""Build a QPDK COMSOL model that holds metal as sheets at the z = 0 interface.

The layout's metal polygons go on a work plane embedded at the silicon/air
interface instead of being extruded, so the unioned model holds two dielectric
domains, air above the plane and silicon below it, and one face per metal
region. That is the representation that meshed and solved cleanly for the QPDK
double-pad transmon; thin extruded solids did not.

Each domain gets a Box selection and a Common material. No physics, mesh, or
study is added: see
:func:`~qpdk.simulation.comsol_capacitance.add_qubit_capacitance_study`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qpdk.simulation.comsol import _add_polygon, _format_number

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol_layout import ComsolLayout

#: Domain selection tags, reused by the physics added on top of this model.
AIR_SELECTION = "air"
SILICON_SELECTION = "si"

#: Relative permittivity of the silicon substrate and of the air above it.
SILICON_RELATIVE_PERMITTIVITY = 11.7
AIR_RELATIVE_PERMITTIVITY = 1.0

#: Half-size in µm of the small boxes that pick one dielectric domain out.
_BOX_HALF_SIZE_UM = 0.01


def _add_block(
    geometry: Any,
    tag: str,
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    """Add one axis-aligned ``Block`` solid anchored at its lower corner."""
    geometry.create(tag, "Block")
    block = geometry.feature(tag)
    block.set("size", [_format_number(value) for value in size])
    block.set("pos", [_format_number(value) for value in origin])


def _add_domain_selection(
    component: Any, tag: str, point: tuple[float, float, float], thickness_um: float
) -> tuple[int, ...]:
    """Add a Box selection that holds the one domain around ``point``."""
    component.selection().create(tag, "Box")
    selection = component.selection(tag)
    # entitydim is a string: an int makes JPype pick the numeric set() overload.
    selection.set("entitydim", "3")
    selection.set("condition", "intersects")
    half_size = min(_BOX_HALF_SIZE_UM, thickness_um / 4.0)
    for axis, value in zip("xyz", point):
        selection.set(f"{axis}min", _format_number(value - half_size))
        selection.set(f"{axis}max", _format_number(value + half_size))
    return tuple(selection.entities())


def _add_material(
    component: Any, tag: str, selection: str, relative_permittivity: float
) -> None:
    """Add a Common material on a named domain selection.

    The permittivity goes into ``propertyGroup("def").set("relpermittivity")``;
    a material has no ``epsilonr`` property for COMSOL to read.
    """
    component.material().create(tag, "Common")
    material = component.material(tag)
    material.selection().named(selection)
    material.propertyGroup("def").set(
        "relpermittivity", _format_number(relative_permittivity)
    )
    material.propertyGroup("def").set("relpermeability", "1")
    material.propertyGroup("def").set("electricconductivity", "0")


def _add_metal_sheets(geometry: Any, layout: ComsolLayout) -> None:
    """Draw the layout's metal polygons on a work plane at the z = 0 interface.

    Each outline becomes a polygon on ``wp1`` and each hole a second polygon
    subtracted by its own ``Difference`` feature. The plane is embedded in the
    geometry, so the polygons become faces, and nothing is extruded.
    """
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


def build_comsol_sheet_model(
    client: mph.Client,
    layout: ComsolLayout,
    name: str,
    *,
    substrate_thickness_um: float = 200.0,
    air_height_um: float = 200.0,
    lateral_margin_um: float = 0.0,
) -> mph.Model:
    """Create a COMSOL 3D model holding the layout's metal as interface sheets.

    The air and silicon blocks touch at ``z = 0`` and span the layout bounding
    box grown by ``lateral_margin_um``. The metal is unioned onto that
    interface, so the model ends up with two dielectric domains and one face per
    metal region.

    Args:
        client: A connected :class:`mph.Client`.
        layout: Extracted metal polygons and bounding box in µm.
        name: Name of the COMSOL model.
        substrate_thickness_um: Silicon thickness below the interface, in µm.
        air_height_um: Air height above the interface, in µm.
        lateral_margin_um: Margin around the layout bounding box for both
            blocks, in µm.

    Returns:
        The MPh model, with geometry, selections, and materials only.

    Raises:
        ValueError: If a thickness is not positive and finite, if the margin is
            negative or not finite, or if the layout has no polygons.
    """
    for label, thickness in (
        ("substrate_thickness_um", substrate_thickness_um),
        ("air_height_um", air_height_um),
    ):
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(f"{label} must be positive and finite, got {thickness!r}")
    if not math.isfinite(lateral_margin_um) or lateral_margin_um < 0.0:
        raise ValueError(
            "lateral_margin_um must be finite and non-negative, "
            f"got {lateral_margin_um!r}"
        )
    if not layout.polygons:
        raise ValueError("layout has no metal polygons to imprint")

    model = client.create(name)
    model.java.component().create("comp1")
    component = model.java.component("comp1")
    geometry = component.geom().create("geom1", 3)
    geometry.lengthUnit("um")

    west, south = (
        layout.bbox.xmin - lateral_margin_um,
        layout.bbox.ymin - lateral_margin_um,
    )
    width = layout.bbox.width + 2.0 * lateral_margin_um
    depth = layout.bbox.height + 2.0 * lateral_margin_um
    _add_block(geometry, "air", (west, south, 0.0), (width, depth, air_height_um))
    _add_block(
        geometry,
        "si",
        (west, south, -substrate_thickness_um),
        (width, depth, substrate_thickness_um),
    )
    _add_metal_sheets(geometry, layout)

    geometry.feature("fin").set("action", "union")
    geometry.run()

    centre = (
        (layout.bbox.xmin + layout.bbox.xmax) / 2.0,
        (layout.bbox.ymin + layout.bbox.ymax) / 2.0,
    )
    air_domains = _add_domain_selection(
        component,
        AIR_SELECTION,
        (centre[0], centre[1], 0.5 * air_height_um),
        air_height_um,
    )
    silicon_domains = _add_domain_selection(
        component,
        SILICON_SELECTION,
        (centre[0], centre[1], -0.5 * substrate_thickness_um),
        substrate_thickness_um,
    )
    if (
        len(air_domains) != 1
        or len(silicon_domains) != 1
        or air_domains == silicon_domains
    ):
        raise ValueError(
            "air and silicon selections must resolve to one distinct domain each, "
            f"got air={air_domains} and silicon={silicon_domains}"
        )

    _add_material(component, "matAir", AIR_SELECTION, AIR_RELATIVE_PERMITTIVITY)
    _add_material(component, "matSi", SILICON_SELECTION, SILICON_RELATIVE_PERMITTIVITY)

    return model
