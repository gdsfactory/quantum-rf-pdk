"""Interdigital capacitor components."""

from __future__ import annotations

from itertools import chain
from math import ceil, floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER


@gf.cell
def interdigital_capacitor(
    fingers: int = 4,
    finger_length: float | int = 20.0,
    finger_gap: float | int = 2.0,
    thickness: float | int = 5.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
    etch_bbox_margin: float = 2.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    """Generate an interdigital capacitor component with ports on both ends.

    An interdigital capacitor consists of interleaved metal fingers that create
    a distributed capacitance. This component creates a planar capacitor with
    two sets of interleaved fingers extending from opposite ends.

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
        layer: Layer to put capacitor geometry on.
        etch_layer: Optional layer for etching around the capacitor.
        etch_bbox_margin: Margin around the capacitor for the etch layer in μm.
        cross_section: Cross-section for the short straight from the etch box capacitor.

    Returns:
        Component: A gdsfactory component with the interdigital capacitor geometry
        and two ports ('o1' and 'o2') on opposing sides.
    """
    c = Component()
    # Used temporarily
    layer = LAYER.M1_DRAW

    if fingers < 1:
        raise ValueError("Must have at least 1 finger")

    width = 2 * thickness + finger_length + finger_gap  # total length
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

    c.add_polygon(points_1, layer=layer)
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
    straight_left = c.add_ref(straight_out_of_etch).move(
        (-etch_bbox_margin, height / 2)
    )
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
    c = gf.boolean(
        A=c,
        B=c_additive,
        operation="A-B",
        layer=etch_layer,
        layer1=etch_layer,
        layer2=layer,
    )
    for port_name, comp in (("o1", straight_left), ("o2", straight_right)):
        c.add_port(name=port_name, port=comp[port_name])

    # Center at (0,0)
    c.move((-width / 2, -height / 2))

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = interdigital_capacitor()
    c.show()
