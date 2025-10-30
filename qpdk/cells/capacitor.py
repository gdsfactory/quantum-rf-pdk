"""Capacitive coupler components."""

from __future__ import annotations

from functools import partial
from itertools import chain
from math import ceil, floor
from operator import itemgetter
from typing import TypedDict, Unpack

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
from qpdk.helper import show_components
from qpdk.tech import LAYER


class InterdigitalCapacitorParams(TypedDict):
    """Parameters for interdigital capacitor.

    Keyword Args:
        fingers: Total number of fingers of the capacitor (must be >= 1).
        finger_length: Length of each finger in μm.
        finger_gap: Gap between adjacent fingers in μm.
        thickness: Thickness of fingers and the base section in μm.
        etch_layer: Optional layer for etching around the capacitor.
        etch_bbox_margin: Margin around the capacitor for the etch layer in μm.
        cross_section: Cross-section for the short straight from the etch box capacitor.
        half: If True, creates a single-sided capacitor (half of the interdigital capacitor).
    """

    fingers: int
    finger_length: float
    finger_gap: float
    thickness: float
    etch_layer: LayerSpec | None
    etch_bbox_margin: float
    cross_section: CrossSectionSpec
    half: bool


_default_interdigital_capacitor_params = InterdigitalCapacitorParams(
    fingers=4,
    finger_length=20.0,
    finger_gap=2.0,
    thickness=5.0,
    etch_layer="M1_ETCH",
    etch_bbox_margin=2.0,
    cross_section="cpw",
    half=False,
)


@gf.cell_with_module_name
def interdigital_capacitor(
    **kwargs: Unpack[InterdigitalCapacitorParams],
) -> Component:
    """Generate an interdigital capacitor component with ports on both ends.

    An interdigital capacitor consists of interleaved metal fingers that create
    a distributed capacitance. This component creates a planar capacitor with
    two sets of interleaved fingers extending from opposite ends.

    .. svgbob::

        ___________                  ___________
       |           |                |           |
       |  o1  _____|    ___    ___  |_____  o2  |
       |     |          | |    | |        |     |
       |     |          | |    | |        |     |
       |     |____  ___  |  ___  |  ____ _|     |
       |__________|                |____________|
              finger_gap     finger_length

    See for example :cite:`leizhuAccurateCircuitModel2000`.

    Note:
        ``finger_length=0`` effectively provides a parallel plate capacitor.
        The capacitance scales approximately linearly with the number of fingers
        and finger length.

    Args:
        kwargs: :class:`~InterdigitalCapacitorParams` for the interdigital capacitor.

    Returns:
        Component: A gdsfactory component with the interdigital capacitor geometry
            and two ports ('o1' and 'o2') on opposing sides.
    """
    c = Component()
    params = _default_interdigital_capacitor_params | kwargs

    (
        fingers,
        finger_length,
        finger_gap,
        thickness,
        etch_layer,
        etch_bbox_margin,
        cross_section,
        half,
    ) = itemgetter(
        "fingers",
        "finger_length",
        "finger_gap",
        "thickness",
        "etch_layer",
        "etch_bbox_margin",
        "cross_section",
        "half",
    )(params)

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
    straight_left = c.add_ref(straight_out_of_etch).move(
        (-etch_bbox_margin, height / 2)
    )
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

    ports_config = [
        ("o1", straight_left["o1"]),
        ("o2", straight_right["o2"]) if not half else None,
    ]

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


@gf.cell_with_module_name
def plate_capacitor(**kwargs: Unpack[InterdigitalCapacitorParams]) -> Component:
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

    .. note::
        This is a special case of the interdigital capacitor with zero finger length.

    Args:
        **kwargs: :class:`~InterdigitalCapacitorParams` for the interdigital

    Returns:
        A gdsfactory component with the plate capacitor geometry.
    """
    return interdigital_capacitor(**(kwargs | {"finger_length": 0}))


@gf.cell_with_module_name
def plate_capacitor_single(**kwargs: Unpack[InterdigitalCapacitorParams]) -> Component:
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
        **kwargs: :class:`~InterdigitalCapacitorParams`

    Returns:
        A gdsfactory component with the plate capacitor geometry.
    """
    return plate_capacitor(**(kwargs | {"half": True}))


if __name__ == "__main__":
    show_components(
        plate_capacitor_single,
        plate_capacitor,
        interdigital_capacitor,
        partial(interdigital_capacitor, half=True),
    )
