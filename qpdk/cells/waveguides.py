"""Waveguide primitives."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Ints, LayerSpec, Size
from kfactory import VInstance
from klayout.db import DCplxTrans

from qpdk import tech
from qpdk.helper import show_components

_DEFAULT_CROSS_SECTION = tech.cpw


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


ring = gf.c.ring

taper_cross_section = partial(
    gf.c.taper_cross_section, cross_section1="cpw", cross_section2="cpw"
)


@gf.cell
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a straight waveguide.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        width: Optional width override in μm.
        npoints: Number of points for the waveguide.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


straight_shorted = straight


@gf.cell
def straight_open(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a straight waveguide with etched gap at one end.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        width: Optional width override in μm.
        npoints: Number of points for the waveguide.
    """
    c = gf.Component()
    straight_ref = c << gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )
    c.add_ports(straight_ref.ports)
    add_etch_gap(c, c.ports["o2"], cross_section=cross_section)
    return c


@gf.cell
def straight_double_open(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    r"""Returns a straight waveguide with etched gaps at both ends.

    Note:
        This may be treated as a :math:`\lambda/2` as a straight resonator in some contexts.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        width: Optional width override in μm.
        npoints: Number of points for the waveguide.
    """
    c = gf.Component()
    straight_ref = c << straight_open(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )
    c.add_ports(straight_ref.ports)
    add_etch_gap(c, c.ports["o1"], cross_section=cross_section)
    return c


@gf.cell
def nxn(
    xsize: float = 10.0,
    ysize: float = 10.0,
    wg_width: float = 10.0,
    layer: LayerSpec = tech.LAYER.M1_DRAW,
    wg_margin: float = 0.0,
    north: int = 1,
    east: int = 1,
    south: int = 1,
    west: int = 1,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
) -> gf.Component:
    """Returns an NxN junction with ports on each side.

    Args:
        xsize: Horizontal size of the junction in μm.
        ysize: Vertical size of the junction in μm.
        wg_width: Width of the waveguides in μm.
        layer: Layer specification.
        wg_margin: Margin from edge to waveguide in μm.
        north: Number of ports on the north side.
        east: Number of ports on the east side.
        south: Number of ports on the south side.
        west: Number of ports on the west side.
        cross_section: Cross-section specification.
    """
    return gf.c.nxn(
        xsize=xsize,
        ysize=ysize,
        wg_width=wg_width,
        layer=layer,
        wg_margin=wg_margin,
        north=north,
        east=east,
        south=south,
        west=west,
        cross_section=cross_section,
    )


@gf.cell
def tee(cross_section: CrossSectionSpec = "cpw") -> gf.Component:
    """Returns a three-way tee waveguide.

    Args:
        cross_section: specification (CrossSection, string or dict).
    """
    c = gf.Component()
    cross_section = gf.get_cross_section(cross_section)
    etch_section = next(
        s
        for s in cross_section.sections
        if s.name is not None and s.name.startswith("etch")
    )
    nxn_ref = c << nxn(
        **{
            "north": 1,
            "east": 1,
            "south": 1,
            "west": 1,
        }
    )
    for port in list(nxn_ref.ports)[:-1]:
        straight_ref = c << straight(
            cross_section=cross_section, length=etch_section.width
        )
        straight_ref.connect("o1", port)

        c.add_port(f"{port.name}", port=straight_ref.ports["o2"])
    etch_ref = c << rectangle(
        size=(etch_section.width, cross_section.width),
        layer=etch_section.layer,
        centered=True,
    )
    etch_ref.transform(
        list(nxn_ref.ports)[-1].dcplx_trans * DCplxTrans(etch_section.width / 2, 0)
    )

    # center
    c.center = (0, 0)

    return c


@gf.cell
def bend_euler(
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
    **kwargs,
) -> gf.Component:
    """Regular degree euler bend.

    Args:
        angle: Angle of the bend in degrees.
        p: Fraction of the bend that is curved (0-1).
        with_arc_floorplan: Include arc floorplan.
        npoints: Number of points for the bend.
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
        **kwargs: Additional arguments passed to gf.c.bend_euler.
    """
    return gf.c.bend_euler(
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


@gf.cell
def bend_circular(
    angle: float = 90.0,
    radius: float = 100.0,
    npoints: int | None = None,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
    allow_min_radius_violation: bool = True,
    **kwargs,
) -> gf.Component:
    """Returns circular bend.

    Args:
        angle: Angle of the bend in degrees.
        radius: Radius of the bend in μm.
        npoints: Number of points for the bend (optional, cannot be used with angular_step).
        cross_section: Cross-section specification.
        width: Optional width override in μm.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
        **kwargs: Additional arguments passed to gf.c.bend_circular (e.g., angular_step).
    """
    return gf.c.bend_circular(
        angle=angle,
        radius=radius,
        npoints=npoints,
        cross_section=cross_section,
        width=width,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


@gf.cell
def bend_s(
    size: Size = (20.0, 3.0),
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
    allow_min_radius_violation: bool = True,
    **kwargs,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: Tuple of (length, offset) for the S bend in μm.
        cross_section: Cross-section specification.
        width: Optional width override in μm.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
        **kwargs: Additional arguments passed to gf.c.bend_s.
    """
    return gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        width=width,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


coupler_straight = partial(gf.c.coupler_straight, cross_section="cpw", gap=16)
coupler_ring = partial(
    gf.c.coupler_ring,
    cross_section="cpw",
    length_x=20,
    bend=bend_circular,
    straight=straight,
    gap=16,
)


@gf.vcell
def straight_all_angle(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    width: float | None = None,
) -> gf.ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: Length of the straight waveguide in μm.
        npoints: Number of points for the waveguide.
        cross_section: Cross-section specification.
        width: Optional width override in μm.

    .. code::

        o1  ──────────────── o2
                length
    """
    return gf.c.straight_all_angle(
        length=length, npoints=npoints, cross_section=cross_section, width=width
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
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
) -> gf.ComponentAllAngle:
    """Returns regular degree euler bend with arbitrary angle.

    Args:
        radius: Radius of the bend in μm.
        angle: Angle of the bend in degrees.
        p: Fraction of the bend that is curved (0-1).
        with_arc_floorplan: Include arc floorplan.
        npoints: Number of points for the bend.
        layer: Layer specification.
        width: Optional width override in μm.
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
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


@gf.vcell
def bend_circular_all_angle(
    radius: float | None = 100.0,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
) -> gf.ComponentAllAngle:
    """Returns circular bend with arbitrary angle.

    Args:
        radius: Radius of the bend in μm.
        angle: Angle of the bend in degrees.
        npoints: Number of points for the bend.
        layer: Layer specification.
        width: Optional width override in μm.
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
    """
    return gf.c.bend_circular_all_angle(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


def add_etch_gap(
    c: gf.Component | gf.ComponentAllAngle,
    port: gf.Port,
    cross_section: CrossSectionSpec,
) -> gf.ComponentReference | VInstance:
    """Adds an etch gap rectangle at the given port of the component.

    Args:
        c: Component to which the etch gap will be added.
        port: Port where the etch gap will be added.
        cross_section: Cross-section specification to determine etch dimensions.
            The etch width is taken from a :class:`~Section` that includes "etch" in its name.
    """
    cross_section = gf.get_cross_section(cross_section)
    etch_section = next(
        s
        for s in cross_section.sections
        if s.name is not None and s.name.startswith("etch")
    )
    etch_ref = c << rectangle(
        size=(etch_section.width, cross_section.width + 2 * etch_section.width),
        layer=etch_section.layer,
        centered=True,
    )
    etch_ref.transform(port.dcplx_trans * DCplxTrans(etch_section.width / 2, 0))
    return etch_ref


if __name__ == "__main__":
    show_components(
        taper_cross_section,
        bend_euler,
        bend_circular,
        tee,
        bend_s,
        straight,
        coupler_ring,
        coupler_straight,
        partial(straight_open, length=20),
        partial(straight_double_open, length=20),
        straight_all_angle,
        partial(bend_euler_all_angle, angle=33),
        rectangle,
        spacing=50,
    )
