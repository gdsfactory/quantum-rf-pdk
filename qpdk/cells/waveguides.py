"""Primitives."""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Ints, LayerSpec, Size


@gf.cell
def rectangle(
    size: Size = (4.0, 2.0),
    layer: LayerSpec = "M1_DRAW",
    centered: bool = False,
    port_type: str | None = "electrical",
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> gf.Component:
    """Returns a rectangle.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0).
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add. None adds no ports.
    """
    c = gf.Component()
    ref = c << gf.c.compass(
        size=size, layer=layer, port_type=port_type, port_orientations=port_orientations
    )
    if not centered:
        ref.move((size[0] / 2, size[1] / 2))
    if port_type:
        c.add_ports(ref.ports)
    c.flatten()
    return c


@gf.cell
def straight(
    length: float = 10,
    cross_section: CrossSectionSpec = "cpw",
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        npoints: number of points.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "cpw",
    allow_min_radius_violation: bool = True,
) -> gf.Component:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return gf.c.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        with_arc_floorplan=True,
        npoints=None,
        layer=None,
    )


@gf.cell
def bend_s(
    size: Size = (11, 1.8),
    cross_section: CrossSectionSpec = "cpw",
    width: float | None = None,
    allow_min_radius_violation: bool = True,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        cross_section: spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        allow_min_radius_violation: allows min radius violations.
    """
    return gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        npoints=99,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.vcell
def straight_all_angle(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "cpw",
    width: float | None = None,
) -> gf.ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

        o1  ──────────────── o2
                length
    """
    return gf.c.straight_all_angle(
        length=length,
        npoints=npoints,
        cross_section=cross_section,
        width=width,
    )


@gf.vcell
def bend_euler_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "cpw",
    allow_min_radius_violation: bool = True,
) -> gf.ComponentAllAngle:
    """Returns regular degree euler bend with arbitrary angle.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: if True the size of the bend will be adjusted to match an arc bend with the specified radius. If False: `radius` is the minimum radius of curvature.
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.

    """
    return gf.c.bend_euler_all_angle(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    c = bend_euler()
    c.pprint_ports()
    c.show()
