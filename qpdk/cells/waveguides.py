"""Waveguide primitives."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Ints, LayerSpec, Size
from kfactory import VInstance
from klayout.db import DCplxTrans

from qpdk import tech
from qpdk.cells._schematic import (
    bend_circular_schematic,
    bend_s_schematic,
    open_schematic,
    short_schematic,
    straight_open_schematic,
    straight_schematic,
    straight_shorted_schematic,
)
from qpdk.logger import logger
from qpdk.tech import get_etch_section

_DEFAULT_CROSS_SECTION = tech.cpw


@gf.cell(tags=("waveguides",))
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
        centered: True sets center to (0.0, 0.0), False sets south-west to (0.0, 0.0).
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


@gf.cell(tags=("waveguides",), schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    npoints: int = 2,
) -> gf.Component:
    """Returns a straight waveguide.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        npoints: Number of points for the waveguide.
    """
    return gf.c.straight(length=length, cross_section=cross_section, npoints=npoints)


straight.schematic_function = straight_schematic


@gf.cell(tags=("terminations",), schematic_function=open_schematic)
def open() -> gf.Component:  # ruff: ignore[builtin-variable-shadowing]
    """Return a layout-free ideal one-port open for schematic simulation."""
    c = gf.Component()
    c.add_port(
        "o1",
        center=(0, 0),
        orientation=180,
        cross_section=_DEFAULT_CROSS_SECTION,
    )
    return c


open.schematic_function = open_schematic


@gf.cell(tags=("terminations",), schematic_function=short_schematic)
def short() -> gf.Component:
    """Return a layout-free ideal one-port short for schematic simulation."""
    c = gf.Component()
    c.add_port(
        "o1",
        center=(0, 0),
        orientation=180,
        cross_section=_DEFAULT_CROSS_SECTION,
    )
    return c


short.schematic_function = short_schematic


@gf.cell(
    tags=("waveguides", "resonators", "terminations"),
    schematic_function=straight_shorted_schematic,
)
def straight_shorted(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    npoints: int = 2,
) -> gf.Component:
    """Return a straight waveguide whose far end is shorted.

    The CPW slots stop at the far end, joining its center trace to ground.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        npoints: Number of points for the waveguide.
    """
    c = gf.Component()
    straight_ref = c << straight(
        length=length, cross_section=cross_section, npoints=npoints
    )
    c.add_port(port=straight_ref.ports["o1"])
    c.add_port(port=straight_ref.ports["o2"], port_type="placement")
    return c


straight_shorted.schematic_function = straight_shorted_schematic


@gf.cell(
    tags=("waveguides", "resonators", "terminations"),
    schematic_function=straight_open_schematic,
)
def straight_open(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    npoints: int = 2,
) -> gf.Component:
    """Returns a straight waveguide with etched gap at one end.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        npoints: Number of points for the waveguide.
    """
    c = gf.Component()
    straight_ref = c << straight(
        length=length, cross_section=cross_section, npoints=npoints
    )
    c.add_port(port=straight_ref.ports["o1"])
    c.add_port(port=straight_ref.ports["o2"], port_type="placement")
    add_etch_gap(c, c.ports["o2"], cross_section=cross_section)
    return c


straight_open.schematic_function = straight_open_schematic


@gf.cell(tags=("waveguides", "resonators"))
def straight_double_open(
    length: float = 10.0,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    npoints: int = 2,
) -> gf.Component:
    r"""Returns a straight waveguide with etched gaps at both ends.

    Note:
        This may be treated as a :math:`\lambda/2` straight resonator in some contexts.

    Args:
        length: Length of the straight waveguide in μm.
        cross_section: Cross-section specification.
        npoints: Number of points for the waveguide.
    """
    c = gf.Component()
    straight_ref = c << straight_open(
        length=length, cross_section=cross_section, npoints=npoints
    )
    c.add_port(port=straight_ref.ports["o1"], port_type="placement")
    c.add_port(port=straight_ref.ports["o2"], port_type="placement")
    add_etch_gap(c, c.ports["o1"], cross_section=cross_section)
    return c


@gf.cell(tags=("waveguides",))
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


@gf.cell(tags=("waveguides",))
def tee(cross_section: CrossSectionSpec = "cpw") -> gf.Component:
    """Returns a three-way tee waveguide.

    Args:
        cross_section: specification (CrossSection, string or dict).
    """
    c = gf.Component()
    cross_section = gf.get_cross_section(cross_section)
    etch_section = get_etch_section(cross_section)
    nxn_ref = c << nxn(**{
        "north": 1,
        "east": 1,
        "south": 1,
        "west": 1,
        "cross_section": cross_section,
        "wg_width": cross_section.width,
        "xsize": cross_section.width,
        "ysize": cross_section.width,
        "layer": cross_section.layer,
    })
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
    c.center = (0.0, 0.0)

    return c


@gf.cell(tags=("waveguides", "bend", "euler"))
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

    Returns:
        The euler bend component.
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


@gf.cell(
    tags=("waveguides", "bend", "circular"), schematic_function=bend_circular_schematic
)
def bend_circular(
    angle: float = 90.0,
    radius: float = 100.0,
    npoints: int | None = None,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
    **kwargs,
) -> gf.Component:
    """Returns circular bend.

    Cross-sections have a minimum value of allowed bend radius, which is half their total width.
    If the user-specified radius is smaller than this value, it is adjusted to the minimum acceptable one.

    Args:
        angle: Angle of the bend in degrees.
        radius: Radius of the bend in μm.
        npoints: Number of points for the bend (optional, cannot be used with angular_step).
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
        **kwargs: Additional arguments passed to gf.c.bend_circular (e.g., angular_step).
    """
    radius_min = gf.get_cross_section(cross_section).radius_min
    if radius_min is not None and radius < radius_min:
        radius = radius_min
        logger.warning(
            (
                "Bend radius needs to be >= {} for this cross-section. "
                "Setting it to the minimum acceptable value."
            ),
            radius_min,
        )
    return gf.c.bend_circular(
        angle=angle,
        radius=radius,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


bend_circular.schematic_function = bend_circular_schematic


@gf.cell(
    tags=("waveguides", "bend", "s"),
    schematic_function=bend_s_schematic,
)
def bend_s(
    size: Size = (20.0, 3.0),
    npoints: int = 99,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
    **kwargs,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: Tuple of (length, offset) for the S bend in μm.
        npoints: Number of points used to discretize the Bézier curve.
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
        **kwargs: Additional arguments passed to gf.c.bend_s.
    """
    return gf.c.bend_s(
        size=size,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


@gf.cell(tags=("waveguides", "couplers"))
def coupler_straight(
    length: float = 10,
    gap: float = 16,
    cross_section: CrossSectionSpec = "cpw",
) -> gf.Component:
    """Return two parallel coupled waveguides.

    Args:
        length: Coupling length in μm.
        gap: Edge-to-edge conductor gap in μm.
        cross_section: Cross-section specification.
    """
    component = gf.Component()
    lower = component << straight(length=length, cross_section=cross_section)
    upper = component << straight(length=length, cross_section=cross_section)
    upper.dmovey(gf.get_cross_section(cross_section).width + gap)
    component.add_ports(lower.ports, prefix="lower_")
    component.add_ports(upper.ports, prefix="upper_")
    component.auto_rename_ports()
    return component


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
) -> gf.ComponentAllAngle:
    """Returns a Straight waveguide with offgrid ports.

    Args:
        length: Length of the straight waveguide in μm.
        npoints: Number of points for the waveguide.
        cross_section: Cross-section specification.

    .. code::

        o1  ──────────────── o2
                length
    """
    return gf.c.straight_all_angle(
        length=length, npoints=npoints, cross_section=cross_section
    )


@gf.vcell
def bend_euler_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
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
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


@gf.vcell
def bend_circular_all_angle(
    radius: float | None = 100.0,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    cross_section: CrossSectionSpec = _DEFAULT_CROSS_SECTION,
    allow_min_radius_violation: bool = True,
) -> gf.ComponentAllAngle:
    """Returns circular bend with arbitrary angle.

    Args:
        radius: Radius of the bend in μm.
        angle: Angle of the bend in degrees.
        npoints: Number of points for the bend.
        layer: Layer specification.
        cross_section: Cross-section specification.
        allow_min_radius_violation: Allow radius smaller than cross-section radius.
    """
    return gf.c.bend_circular_all_angle(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
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

    Returns:
        Reference or VInstance of the added etch gap.
    """
    cross_section = gf.get_cross_section(cross_section)
    etch_section = get_etch_section(cross_section)
    etch_ref = c << rectangle(
        size=(etch_section.width, cross_section.width + 2 * etch_section.width),
        layer=etch_section.layer,
        centered=True,
    )
    etch_ref.transform(port.dcplx_trans * DCplxTrans(etch_section.width / 2, 0))
    return etch_ref
