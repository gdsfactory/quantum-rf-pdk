"""Primitives."""

from functools import partial
from typing import TypedDict, Unpack

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Ints, LayerSpec, Size
from klayout.db import DCplxTrans

from qpdk import tech
from qpdk.helper import show_components

_DEFAULT_CROSS_SECTION = tech.cpw
_DEFAULT_KWARGS = {"cross_section": _DEFAULT_CROSS_SECTION}
_DEFAULT_BEND_KWARGS = _DEFAULT_KWARGS | {"allow_min_radius_violation": True}


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


class StraightKwargs(TypedDict, total=False):
    """Type definition for straight keyword arguments."""

    length: float
    cross_section: CrossSectionSpec
    width: float | None
    npoints: int


@gf.cell
def straight(**kwargs: Unpack[StraightKwargs]) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        **kwargs: Arguments passed to gf.c.straight.
    """
    return gf.c.straight(**(_DEFAULT_KWARGS | kwargs))


class NxnKwargs(TypedDict, total=False):
    """Type definition for tee keyword arguments."""

    xsize: float
    ysize: float
    wg_width: float
    layer: LayerSpec
    wg_margin: float
    north: int
    east: int
    south: int
    west: int


_NXN_DEFAULTS = {
    "xsize": 10.0,
    "ysize": 10.0,
    "wg_width": 10,
    "layer": tech.LAYER.M1_DRAW,
    "wg_margin": 0,
    "north": 1,
    "east": 1,
    "south": 1,
    "west": 1,
}


@gf.cell
def nxn(**kwargs: Unpack[NxnKwargs]) -> gf.Component:
    """Returns a tee waveguide.

    Args:
        **kwargs: Arguments passed to gf.c.nxn.
    """
    return gf.c.nxn(**(_DEFAULT_KWARGS | _NXN_DEFAULTS | kwargs))


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


class BendEulerKwargs(TypedDict, total=False):
    """Type definition for bend_euler keyword arguments."""

    angle: float
    p: float
    with_arc_floorplan: bool
    npoints: int
    direction: str
    with_cladding_box: bool
    cross_section: gf.CrossSection
    allow_min_radius_violation: bool


@gf.cell
def bend_euler(**kwargs: Unpack[BendEulerKwargs]) -> gf.Component:
    """Regular degree euler bend.

    Args:
        **kwargs: Arguments passed to gf.c.bend_euler.
    """
    return gf.c.bend_euler(**(_DEFAULT_BEND_KWARGS | kwargs))


class BendCircularKwargs(TypedDict, total=False):
    """Type definition for bend_circular keyword arguments."""

    angle: float
    npoints: int
    with_arc_floorplan: bool
    cross_section: gf.CrossSection
    radius: float
    direction: str
    allow_min_radius_violation: bool


_BEND_CIRCULAR_DEFAULTS = {
    "radius": 100,
}


@gf.cell
def bend_circular(**kwargs: Unpack[BendCircularKwargs]) -> gf.Component:
    """Returns circular bend.

    Args:
        **kwargs: Arguments passed to gf.c.bend_circular.
    """
    return gf.c.bend_circular(
        **(_DEFAULT_BEND_KWARGS | _BEND_CIRCULAR_DEFAULTS | kwargs)
    )


class BendSKwargs(TypedDict, total=False):
    """Type definition for bend_s keyword arguments."""

    size: Size
    cross_section: CrossSectionSpec
    width: float | None
    allow_min_radius_violation: bool


_BEND_S_DEFAULTS = {
    "size": (20.0, 3.0),
}


@gf.cell
def bend_s(**kwargs: Unpack[BendSKwargs]) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        **kwargs: Arguments passed to gf.c.bend_s.
    """
    return gf.c.bend_s(**(_DEFAULT_BEND_KWARGS | _BEND_S_DEFAULTS | kwargs))


class StraightAllAngleKwargs(TypedDict, total=False):
    """Type definition for straight_all_angle keyword arguments."""

    length: float
    npoints: int
    cross_section: CrossSectionSpec
    width: float | None


@gf.vcell
def straight_all_angle(
    **kwargs: Unpack[StraightAllAngleKwargs],
) -> gf.ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        **kwargs: Arguments passed to gf.c.straight_all_angle.

    .. code::

        o1  ──────────────── o2
                length
    """
    return gf.c.straight_all_angle(**(_DEFAULT_KWARGS | kwargs))


class BendEulerAllAngleKwargs(TypedDict, total=False):
    """Type definition for bend_euler_all_angle keyword arguments."""

    radius: float | None
    angle: float
    p: float
    with_arc_floorplan: bool
    npoints: int | None
    layer: gf.typings.LayerSpec | None
    width: float | None
    cross_section: CrossSectionSpec
    allow_min_radius_violation: bool


@gf.vcell
def bend_euler_all_angle(
    **kwargs: Unpack[BendEulerAllAngleKwargs],
) -> gf.ComponentAllAngle:
    """Returns regular degree euler bend with arbitrary angle.

    Args:
        **kwargs: Arguments passed to gf.c.bend_euler_all_angle.
    """
    return gf.c.bend_euler_all_angle(**(_DEFAULT_BEND_KWARGS | kwargs))


class BendCircularAllAngleKwargs(TypedDict, total=False):
    """Type definition for bend_circular_all_angle keyword arguments."""

    radius: float | None
    angle: float
    npoints: int | None
    layer: gf.typings.LayerSpec | None
    width: float | None
    cross_section: CrossSectionSpec
    allow_min_radius_violation: bool


@gf.vcell
def bend_circular_all_angle(
    **kwargs: Unpack[BendCircularAllAngleKwargs],
) -> gf.ComponentAllAngle:
    """Returns circular bend with arbitrary angle.

    Args:
        **kwargs: Arguments passed to gf.c.bend_circular_all_angle.
    """
    return gf.c.bend_circular_all_angle(
        **(_DEFAULT_BEND_KWARGS | _BEND_CIRCULAR_DEFAULTS | kwargs)
    )


if __name__ == "__main__":
    show_components(
        taper_cross_section,
        bend_euler,
        bend_circular,
        tee,
        bend_s,
        straight,
        straight_all_angle,
        partial(bend_euler_all_angle, angle=33),
        rectangle,
    )
