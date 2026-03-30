"""Capacitive coupler components."""

from __future__ import annotations

from functools import partial
from itertools import chain
from math import ceil, floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import add_etch_gap, bend_circular, straight
from qpdk.helper import show_components
from qpdk.tech import LAYER, get_etch_section, get_etch_sections


@gf.cell
def half_circle_coupler(
    radius: float = 50.0,
    angle: float = 180.0,
    extension_length: float = 10.0,
    cross_section: CrossSectionSpec = "cpw",
    extra_straight_length: float = 20.0,
    open_end: bool = True,
) -> Component:
    """Creates a half-circle coupler for readout.

    This coupler consists of a circular bend (typically 180 degrees) that wraps
    around a resonator arm for capacitive coupling.

    Args:
        radius: Inner radius of the half-circle in μm.
        angle: Angle of the circular arc in degrees.
        extension_length: Length of the straight sections extending from the
            ends of the half-circle in μm.
        cross_section: Cross-section specification for the coupler.
        extra_straight_length: Length of the straight section extending from the
            bottom of the half-circle in μm.
        open_end: If True, adds an etched gap at the ends of the extensions.

    Returns:
        Component: A gdsfactory component with the half-circle coupler geometry.
    """
    c = Component()

    bend = c.add_ref(
        bend_circular(
            radius=radius,
            angle=angle,
            cross_section=cross_section,
        )
    )

    # Position bend such that it's centered and opening upwards
    bend.rotate(-angle / 2)
    bend.move((-bend.dcenter[0], -bend.dcenter[1]))

    # Add extensions to the ends of the bend
    if extension_length > 0:
        ext1 = c.add_ref(straight(length=extension_length, cross_section=cross_section))
        ext1.connect("o1", bend.ports["o1"])
        ext2 = c.add_ref(straight(length=extension_length, cross_section=cross_section))
        ext2.connect("o1", bend.ports["o2"])
        where_to_add_gaps = [ext1.ports["o2"], ext2.ports["o2"]]
    else:
        where_to_add_gaps = [bend.ports["o1"], bend.ports["o2"]]

    if open_end:
        for port in where_to_add_gaps:
            add_etch_gap(c, port, cross_section=cross_section)

    # Get cross section details to calculate overlap
    xs = gf.get_cross_section(cross_section)
    cross_section_etch_section = get_etch_section(xs)
    # Ensure significant overlap by moving stem into the bend metal
    # and considering the bend radius
    overlap = xs.width / 2 + cross_section_etch_section.width

    # Add a stem/lead straight from the center of the arc.
    # The stem uses the CPW cross-section but does NOT extend into the bend,
    # to avoid M1_ETCH overlapping with the bend's M1_DRAW.
    stem = c.add_ref(
        straight(
            length=extra_straight_length,
            cross_section=cross_section,
        )
    )

    stem.rotate(-90)
    stem.movey(bend.dbbox().bottom)
    stem.movex(bend.dcenter[0])  # Center it

    # Bridge the stem into the bend with M1_DRAW center conductor and
    # M1_ETCH gap sections, matching the CPW cross-section pattern.
    # Unlike using a straight ref, these polygons avoid hierarchical
    # M1_ETCH-over-M1_DRAW conflicts with the bend.
    bridge_x = bend.dcenter[0]
    bridge_y_bottom = bend.dbbox().bottom
    bridge_y_top = bridge_y_bottom + overlap
    c.add_polygon(
        [
            (bridge_x - xs.width / 2, bridge_y_bottom),
            (bridge_x + xs.width / 2, bridge_y_bottom),
            (bridge_x + xs.width / 2, bridge_y_top),
            (bridge_x - xs.width / 2, bridge_y_top),
        ],
        layer=LAYER.M1_DRAW,
    )
    etch_height = overlap / 3
    for etch_s in get_etch_sections(xs):
        etch_x_center = bridge_x + etch_s.offset
        c.add_polygon(
            [
                (etch_x_center - etch_s.width / 2, bridge_y_bottom),
                (etch_x_center + etch_s.width / 2, bridge_y_bottom),
                (etch_x_center + etch_s.width / 2, bridge_y_bottom + etch_height),
                (etch_x_center - etch_s.width / 2, bridge_y_bottom + etch_height),
            ],
            layer=LAYER.M1_ETCH,
        )
    c.add_port("o3", port=stem.ports["o2"])

    # Place anchor at the arc center, computed as the midpoint of the
    # bend ports for a circular arc, for concentric alignment with an
    # inner resonator bend.
    arc_center_x = (bend.ports["o1"].dx + bend.ports["o2"].dx) / 2
    arc_center_y = (bend.ports["o1"].dy + bend.ports["o2"].dy) / 2
    c.add_port(
        name="anchor",
        center=(arc_center_x, arc_center_y),
        width=xs.width,
        orientation=90,
        layer=LAYER.M1_DRAW,
        port_type="placement",
    )

    return c


@gf.cell
def interdigital_capacitor(
    fingers: int = 4,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
    etch_bbox_margin: float = 2.0,
    cross_section: CrossSectionSpec = "cpw",
    half: bool = False,
) -> Component:
    """Generate an interdigital capacitor component with ports on both ends.

    An interdigital capacitor consists of interleaved metal fingers that create
    a distributed capacitance. This component creates a planar capacitor with
    two sets of interleaved fingers extending from opposite ends.

    .. svgbob::

          ┌─┐───────┐┌─┐
          │ │───────┘│ │
          │ │ ┌──────│ │
         ┌│ │ └──────│ │┐
       o1└│ │──────┐ │ │┘o2
          │ │──────┘ │ │
          │ │ ┌──────│ │
          └─┘ └──────└─┘

    See for example :cite:`leizhuAccurateCircuitModel2000`.

    Note:
        ``finger_length=0`` effectively provides a parallel plate capacitor.
        The capacitance scales approximately linearly with the number of fingers
        and finger length.

    Args:
        fingers: Total number of fingers of the capacitor (must be >= 1).
        finger_length: Length of each finger in μm.
        finger_gap: Gap between adjacent fingers in μm.
        thickness: Thickness of fingers and the base section in μm.
        etch_layer: Optional layer for etching around the capacitor.
        etch_bbox_margin: Margin around the capacitor for the etch layer in μm.
        cross_section: Cross-section for the short straight from the etch box capacitor.
        half: If True, creates a single-sided capacitor (half of the interdigital capacitor).

    Returns:
        Component: A gdsfactory component with the interdigital capacitor geometry
            and two ports ('o1' and 'o2') on opposing sides.

    Raises:
        ValueError: If fingers is less than 1.
    """
    c = Component()

    # Used temporarily
    layer = LAYER.M1_DRAW

    if fingers < 1:
        raise ValueError("Must have at least 1 finger")

    width = (
        2 * thickness + finger_length + finger_gap
        if not half
        else thickness + finger_length
    )  # total length
    height = fingers * thickness + (fingers - 1) * finger_gap  # total height
    points_1 = [
        (0, 0),
        (0, height),
        (thickness + finger_length, height),
        (thickness + finger_length, height - thickness),
        (thickness, height - thickness),
        *chain.from_iterable(
            (
                (thickness, height - (2 * i) * (thickness + finger_gap)),
                (
                    thickness + finger_length,
                    height - (2 * i) * (thickness + finger_gap),
                ),
                (
                    thickness + finger_length,
                    height - (2 * i) * (thickness + finger_gap) - thickness,
                ),
                (thickness, height - (2 * i) * (thickness + finger_gap) - thickness),
            )
            for i in range(ceil(fingers / 2))
        ),
        (thickness, 0),
        (0, 0),
    ]
    c.add_polygon(points_1, layer=layer)

    if not half:
        points_2 = [
            (width, 0),
            (width, height),
            (width - thickness, height),
            *chain.from_iterable(
                (
                    (
                        width - thickness,
                        height - (1 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                    ),
                    (
                        width - (thickness + finger_length),
                        height - (1 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                    ),
                    (
                        width - (thickness + finger_length),
                        height - (2 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                    ),
                    (
                        width - thickness,
                        height - (2 + 2 * i) * thickness - (1 + 2 * i) * finger_gap,
                    ),
                )
                for i in range(floor(fingers / 2))
            ),
            (width - thickness, 0),
            (width, 0),
        ]
        c.add_polygon(points_2, layer=layer)

    # Add etch layer bbox if specified
    if etch_layer is not None:
        etch_bbox = [
            (-etch_bbox_margin, -etch_bbox_margin),
            (width + etch_bbox_margin, -etch_bbox_margin),
            (width + etch_bbox_margin, height + etch_bbox_margin),
            (-etch_bbox_margin, height + etch_bbox_margin),
        ]
        c.add_polygon(etch_bbox, layer=etch_layer)

    # Add small straights on the left and right sides of the capacitor
    straight_cross_section = gf.get_cross_section(cross_section)
    straight_out_of_etch = straight(
        length=etch_bbox_margin, cross_section=straight_cross_section
    )
    straight_left = c.add_ref(straight_out_of_etch).move((
        -etch_bbox_margin,
        height / 2,
    ))
    straight_right = None
    if not half:
        straight_right = c.add_ref(straight_out_of_etch).move((width, height / 2))

    # Add WG to additive metal
    c_additive = gf.boolean(
        A=c,
        B=c,
        operation="or",
        layer=layer,
        layer1=layer,
        layer2=straight_cross_section.layer,
    )

    # Take boolean negative
    c_negative = gf.boolean(
        A=c,
        B=c_additive,
        operation="A-B",
        layer=etch_layer,
        layer1=etch_layer,
        layer2=layer,
    )

    # Combine results
    c = gf.Component()
    c.absorb(c << c_additive)
    c.absorb(c << c_negative)

    ports_config: list[tuple[str, gf.Port] | None] = [
        ("o1", straight_left["o1"]),
    ]
    if not half and straight_right is not None:
        ports_config.append(("o2", straight_right["o2"]))

    for port_name, port_ref in filter(None, ports_config):
        c.add_port(
            name=port_name,
            width=port_ref.width,
            center=port_ref.center,
            orientation=port_ref.orientation,
            layer=LAYER.M1_DRAW,
        )

    # Center at (0,0)
    c.move((-width / 2, -height / 2))

    return c


@gf.cell
def plate_capacitor(
    length: float = 26.0,
    width: float = 5.0,
    gap: float = 7.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
    etch_bbox_margin: float = 2.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    """Creates a plate capacitor.

    A capacitive coupler consists of two metal pads separated by a small gap,
    providing capacitive coupling between circuit elements like qubits and resonators.

    .. svgbob::

                  ______               ______
        _________|      |             |      |________
       |                |             |               |
       |  o1       pad1 | ====gap==== | pad2      o2  |
       |                |             |               |
       |_________       |             |      _________|
                 |______|             |______|

    Args:
        length: Length (vertical extent) of the capacitor pad in μm.
        width: Width (horizontal extent) of the capacitor pad in μm.
        gap: Gap between plates in μm.
        etch_layer: Optional layer for etching around the capacitor.
        etch_bbox_margin: Margin around the capacitor for the etch layer in μm.
        cross_section: Cross-section for the short straight from the etch box capacitor.

    Returns:
        A gdsfactory component with the plate capacitor geometry and two ports ('o1' and 'o2') on opposing sides.

    Raises:
        ValueError: If width or length is not positive.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    c = Component()
    single_capacitor = plate_capacitor_single(
        length=length,
        width=width,
        etch_layer=etch_layer,
        etch_bbox_margin=etch_bbox_margin,
        cross_section=cross_section,
    )

    pad1 = c.add_ref(single_capacitor)
    pad2 = c.add_ref(single_capacitor)
    pad2.rotate(180)
    pad2.move((width + gap, 0))
    c.center = (0, 0)

    # Add ports
    c.add_port(name="o1", port=pad1.ports["o1"])
    c.add_port(name="o2", port=pad2.ports["o1"])

    # Ensure etch box between pads
    if etch_layer is not None:
        missing_width = gap - 2 * etch_bbox_margin
        if missing_width > 0:
            etch_bbox = [
                (-missing_width / 2, -length / 2 - etch_bbox_margin),
                (missing_width / 2, -length / 2 - etch_bbox_margin),
                (missing_width / 2, length / 2 + etch_bbox_margin),
                (-missing_width / 2, length / 2 + etch_bbox_margin),
            ]
            c.add_polygon(etch_bbox, layer=etch_layer)

    return c


@gf.cell
def plate_capacitor_single(
    length: float = 26.0,
    width: float = 5.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
    etch_bbox_margin: float = 2.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    """Creates a single plate capacitor for coupling.

    This is essentially half of a :func:`~plate_capacitor`.

    .. svgbob::

                  ______
        _________|      |
       |                |
       |  o1       pad1 |
       |                |
       |_________       |
                 |______|

    Args:
        length: Length (vertical extent) of the capacitor pad in μm.
        width: Width (horizontal extent) of the capacitor pad in μm.
        etch_layer: Optional layer for etching around the capacitor.
        etch_bbox_margin: Margin around the capacitor for the etch layer in μm.
        cross_section: Cross-section for the short straight from the etch box capacitor.

    Returns:
        A gdsfactory component with the plate capacitor geometry.

    Raises:
        ValueError: If width or length is not positive.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    c = Component()

    layer = LAYER.M1_DRAW
    points = [
        (0, 0),
        (0, length),
        (width, length),
        (width, 0),
    ]
    c.add_polygon(points, layer=layer)
    # Add etch layer bbox if specified
    if etch_layer is not None:
        etch_bbox = [
            (-etch_bbox_margin, -etch_bbox_margin),
            (width + etch_bbox_margin, -etch_bbox_margin),
            (width + etch_bbox_margin, length + etch_bbox_margin),
            (-etch_bbox_margin, length + etch_bbox_margin),
        ]
        c.add_polygon(etch_bbox, layer=etch_layer)
    # Add small straight on the left side of the capacitor
    straight_cross_section = gf.get_cross_section(cross_section)
    straight_out_of_etch = straight(
        length=etch_bbox_margin, cross_section=straight_cross_section
    )
    straight_left = c.add_ref(straight_out_of_etch).move((
        -etch_bbox_margin,
        length / 2,
    ))
    # Add WG to additive metal
    c_additive = gf.boolean(
        A=c,
        B=c,
        operation="or",
        layer=layer,
        layer1=layer,
        layer2=straight_cross_section.layer,
    )

    # Take boolean negative
    c_negative = gf.boolean(
        A=c,
        B=c_additive,
        operation="A-B",
        layer=etch_layer,
        layer1=etch_layer,
        layer2=layer,
    )
    # Combine results
    c = gf.Component()
    c.absorb(c << c_additive)
    c.absorb(c << c_negative)

    c.add_port(
        name="o1",
        width=straight_left["o1"].width,
        center=straight_left["o1"].center,
        orientation=straight_left["o1"].orientation,
        layer=LAYER.M1_DRAW,
    )

    # Center at (0,0)
    c.move((-width / 2, -length / 2))

    return c


if __name__ == "__main__":
    show_components(
        half_circle_coupler,
        plate_capacitor_single,
        plate_capacitor,
        interdigital_capacitor,
        partial(interdigital_capacitor, half=True),
    )
