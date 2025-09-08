"""Capacitive coupler components."""

from __future__ import annotations

from itertools import chain
from math import ceil, floor
from typing import TypedDict, Unpack

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
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
    """

    fingers: int
    finger_length: float
    finger_gap: float
    thickness: float
    etch_layer: LayerSpec | None
    etch_bbox_margin: float
    cross_section: CrossSectionSpec


_default_interdigital_capacitor_params = InterdigitalCapacitorParams(
    fingers=4,
    finger_length=20.0,
    finger_gap=2.0,
    thickness=5.0,
    etch_layer="M1_ETCH",
    etch_bbox_margin=2.0,
    cross_section="cpw",
)


@gf.cell_with_module_name
def interdigital_capacitor(
    **kwargs: Unpack[InterdigitalCapacitorParams],
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
    ) = (
        params[key]
        for key in [
            "fingers",
            "finger_length",
            "finger_gap",
            "thickness",
            "etch_layer",
            "etch_bbox_margin",
            "cross_section",
        ]
    )
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


@gf.cell_with_module_name
def plate_capacitor(**kwargs: Unpack[InterdigitalCapacitorParams]) -> Component:
    """Creates a plate capacitor.

    A capacitive coupler consists of two metal pads separated by a small gap,
    providing capacitive coupling between circuit elements like qubits and resonators.

    .. code::
                    ______               ______
          _______  |      |             |      | _______
         |       | |      |             |      ||       |
         |  o1   | | pad1 | ====gap==== | pad2 ||   o2  |
         |       | |      |             |      ||       |
         |_______| |      |             |      ||_______|
                   |______|             |______|

    Note:
        This is a special case of the interdigital capacitor with zero finger length.

    Args:
        **kwargs: :class:`~InterdigitalCapacitorParams` for the interdigital

    Returns:
        Component: A gdsfactory component with the plate capacitor geometry.
    """
    return interdigital_capacitor(**(kwargs | {"finger_length": 0}))


@gf.cell_with_module_name
def coupler_tunable(
    pad_width: float = 30.0,
    pad_height: float = 40.0,
    gap: float = 3.0,
    tuning_pad_width: float = 15.0,
    tuning_pad_height: float = 20.0,
    tuning_gap: float = 1.0,
    feed_width: float = 10.0,
    feed_length: float = 30.0,
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    port_type: str = "electrical",
) -> Component:
    """Creates a tunable capacitive coupler with voltage control.

    A tunable coupler includes additional electrodes that can be voltage-biased
    to change the coupling strength dynamically.


    Args:
        pad_width: Width of main coupling pads in μm.
        pad_height: Height of main coupling pads in μm.
        gap: Gap between main coupling pads in μm.
        tuning_pad_width: Width of tuning pads in μm.
        tuning_pad_height: Height of tuning pads in μm.
        tuning_gap: Gap to tuning pads in μm.
        feed_width: Width of feed lines in μm.
        feed_length: Length of feed lines in μm.
        layer_metal: Layer for main metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the tunable coupler geometry.


    .. code::

                    (connected to feed)
                         _______
                        |       |
                        | tpad1 |
                        |       |
                        |_______|
                        tuning gap
                   ______        ______
         _______  |      |      |      | _______
        |       | |      |      |      ||       |
        | feed1 | | pad1 | gap  | pad2 || feed2 |
        |       | |      |      |      ||       |
        |_______| |      |      |      ||_______|
                  |______|      |______|
                        tuning gap
                         _______
                        |       |
                        | tpad2 |
                        |       |
                        |_______|
                    (connected to feed)
    """
    c = Component()

    # Create main coupling pads
    left_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    left_pad_ref = c.add_ref(left_pad)
    left_pad_ref.move((-pad_width - gap / 2, -pad_height / 2))

    right_pad = gf.components.rectangle(
        size=(pad_width, pad_height),
        layer=layer_metal,
    )
    right_pad_ref = c.add_ref(right_pad)
    right_pad_ref.move((gap / 2, -pad_height / 2))

    # Create tuning pads above and below
    top_tuning_pad = gf.components.rectangle(
        size=(tuning_pad_width, tuning_pad_height),
        layer=layer_metal,
    )
    top_tuning_ref = c.add_ref(top_tuning_pad)
    top_tuning_ref.move((-tuning_pad_width / 2, pad_height / 2 + tuning_gap))

    bottom_tuning_pad = gf.components.rectangle(
        size=(tuning_pad_width, tuning_pad_height),
        layer=layer_metal,
    )
    bottom_tuning_ref = c.add_ref(bottom_tuning_pad)
    bottom_tuning_ref.move(
        (-tuning_pad_width / 2, -pad_height / 2 - tuning_gap - tuning_pad_height)
    )

    # Create feed lines for main pads
    left_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    left_feed_ref = c.add_ref(left_feed)
    left_feed_ref.move((-pad_width - gap / 2 - feed_length, -feed_width / 2))

    right_feed = gf.components.rectangle(
        size=(feed_length, feed_width),
        layer=layer_metal,
    )
    right_feed_ref = c.add_ref(right_feed)
    right_feed_ref.move((gap / 2 + pad_width, -feed_width / 2))

    # Create tuning feed lines
    top_tuning_feed = gf.components.rectangle(
        size=(feed_width, feed_length),
        layer=layer_metal,
    )
    top_tuning_feed_ref = c.add_ref(top_tuning_feed)
    top_tuning_feed_ref.move(
        (-feed_width / 2, pad_height / 2 + tuning_gap + tuning_pad_height)
    )

    bottom_tuning_feed = gf.components.rectangle(
        size=(feed_width, feed_length),
        layer=layer_metal,
    )
    bottom_tuning_feed_ref = c.add_ref(bottom_tuning_feed)
    bottom_tuning_feed_ref.move(
        (
            -feed_width / 2,
            -pad_height / 2 - tuning_gap - tuning_pad_height - feed_length,
        )
    )

    # Add ports
    c.add_port(
        name="left",
        center=(-pad_width - gap / 2 - feed_length, 0),
        width=feed_width,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="right",
        center=(gap / 2 + pad_width + feed_length, 0),
        width=feed_width,
        orientation=0,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="tuning_top",
        center=(0, pad_height / 2 + tuning_gap + tuning_pad_height + feed_length),
        width=feed_width,
        orientation=90,
        layer=layer_metal,
        port_type=port_type,
    )

    c.add_port(
        name="tuning_bottom",
        center=(0, -pad_height / 2 - tuning_gap - tuning_pad_height - feed_length),
        width=feed_width,
        orientation=270,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["coupler_type"] = "tunable"
    c.info["pad_width"] = pad_width
    c.info["pad_height"] = pad_height
    c.info["gap"] = gap
    c.info["tuning_pad_width"] = tuning_pad_width
    c.info["tuning_pad_height"] = tuning_pad_height
    c.info["tuning_gap"] = tuning_gap

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = gf.Component()
    for i, component in enumerate(
        (
            plate_capacitor(),
            coupler_tunable(),
            interdigital_capacitor(),
        )
    ):
        (c << component).move((i * 200, 0))
    c.pprint_ports()
    c.show()
