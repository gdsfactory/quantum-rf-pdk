"""Inductor and lumped-element resonator components."""

from __future__ import annotations

from math import ceil, floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER, coplanar_waveguide, xsection


@xsection
def default_meander_inductor_cross_section() -> CrossSectionSpec:
    """Default cross-section for the meander inductor."""
    return coplanar_waveguide(
        width=2.0,
        gap=2.0,
    )

@gf.cell
def meander_inductor(
    n_turns: int = 5,
    turn_length: float = 200.0,
    cross_section: CrossSectionSpec = default_meander_inductor_cross_section,
    etch_bbox_margin: float = 2.0,
) -> Component:
    r"""Creates a meander inductor with Manhattan routing using a narrow wire.

    The inductor consists of multiple horizontal runs connected by short
    vertical segments at alternating ends, forming a serpentine (meander)
    path.  The total inductance is dominated by kinetic inductance for
    superconducting thin films.

    .. svgbob::

        o1 ─────────────────────┐
                                │
        ┌───────────────────────┘
        │
        └───────────────────────┐
                                │
        ┌───────────────────────┘
        │
        └────────────────────── o2

    Similar structures are described in
    :cite:`kimThinfilmSuperconductingResonator2011,chenCompactInductorcapacitorResonators2023`.

    Args:
        n_turns: Number of horizontal meander runs (must be >= 1).
        turn_length: Length of each horizontal run in µm.
        cross_section: Cross-section specification for the meander wire.
            The center conductor width and etch gap are derived from this
            specification. The meander's vertical pitch is set to ensure that
            the etched regions of adjacent runs do not overlap, maintaining
            the characteristic impedance of each run. Specifically, the pitch
            is calculated as :math:`w + 2g`, where :math:`w` is the wire width
            and :math:`g` is the etch gap.
        etch_bbox_margin: Extra margin around the inductor for the etch bounding box in µm.
            This margin is added in addition to the etch region defined in the cross-section.

    Returns:
        Component: A gdsfactory component with the meander inductor geometry
            and two ports ('o1' and 'o2').
    """
    if n_turns < 1:
        raise ValueError("Must have at least 1 turn")
    if turn_length <= 0:
        raise ValueError(f"turn_length must be positive, got {turn_length}")

    xs = gf.get_cross_section(cross_section)
    wire_width = xs.width
    layer = xs.layer

    # Infer etch parameters and spacing from cross section
    try:
        etch_section = next(
            s for s in xs.sections if s.name and "etch_offset" in s.name
        )
    except StopIteration as e:
        raise ValueError(
            f"Cross-section '{xs.name}' does not have a section with 'etch_offset' in the name. "
            "The `meander_inductor` requires a cross-section with at least one etch section "
            "(e.g., 'coplanar_waveguide') to correctly determine the meander pitch and "
            "bounding box margin."
        ) from e

    etch_layer = etch_section.layer
    etch_width = etch_section.width
    # For CPW-like structures, we assume a pitch that allows for non-overlapping etches
    # i.e. pitch = width + 2 * gap, which means wire_gap = 2 * etch_width
    wire_gap = 2 * etch_width

    c = Component()
    pitch = wire_width + wire_gap
    total_height = n_turns * wire_width + max(0, n_turns - 1) * wire_gap

    for i in range(n_turns):
        y0 = i * pitch
        c.add_polygon(
            [
                (0, y0),
                (turn_length, y0),
                (turn_length, y0 + wire_width),
                (0, y0 + wire_width),
            ],
            layer=layer,
        )

    for i in range(n_turns - 1):
        y0 = i * pitch + wire_width
        y1 = (i + 1) * pitch
        if i % 2 == 0:
            c.add_polygon(
                [
                    (turn_length - wire_width, y0),
                    (turn_length, y0),
                    (turn_length, y1),
                    (turn_length - wire_width, y1),
                ],
                layer=layer,
            )
        else:
            c.add_polygon(
                [(0, y0), (wire_width, y0), (wire_width, y1), (0, y1)],
                layer=layer,
            )

    if etch_layer is not None:
        # Extra margin on top of the implicit etch margin from the cross-section
        margin = etch_width + etch_bbox_margin
        c.add_polygon(
            [
                (-margin, -margin),
                (turn_length + margin, -margin),
                (turn_length + margin, total_height + margin),
                (-margin, total_height + margin),
            ],
            layer=etch_layer,
        )

        c_metal = gf.boolean(
            A=c, B=c, operation="or", layer=layer, layer1=layer, layer2=layer
        )
        c_etch = gf.boolean(
            A=c,
            B=c_metal,
            operation="A-B",
            layer=etch_layer,
            layer1=etch_layer,
            layer2=layer,
        )
        c = gf.Component()
        c.absorb(c << c_metal)
        c.absorb(c << c_etch)

    c.add_port(
        name="o1",
        center=(0, wire_width / 2),
        width=wire_width,
        orientation=180,
        layer=layer,
        port_type="electrical",
        cross_section=xs,
    )

    last_run_center_y = (n_turns - 1) * pitch + wire_width / 2
    if n_turns % 2 == 1:
        c.add_port(
            name="o2",
            center=(turn_length, last_run_center_y),
            width=wire_width,
            orientation=0,
            layer=layer,
            port_type="electrical",
            cross_section=xs,
        )
    else:
        c.add_port(
            name="o2",
            center=(0, last_run_center_y),
            width=wire_width,
            orientation=180,
            layer=layer,
            port_type="electrical",
            cross_section=xs,
        )

    c.move((-turn_length / 2, -total_height / 2))

    total_wire_length = n_turns * turn_length + max(0, n_turns - 1) * wire_gap
    c.info["total_wire_length"] = total_wire_length
    c.info["n_squares"] = total_wire_length / wire_width
    c.info["cross_section"] = xs.name

    return c



@gf.cell
def lumped_element_resonator(
    fingers: int = 20,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    finger_thickness: float = 5.0,
    n_turns: int = 15,
    wire_width: float = 2.0,
    wire_gap: float = 4.0,
    bus_bar_spacing: float = 4.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
    etch_bbox_margin: float = 2.0,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    r"""Creates a lumped-element resonator combining an interdigital capacitor and a meander inductor.

    The resonator consists of an interdigital capacitor section (providing
    capacitance) connected in parallel with a meander inductor section
    (providing inductance) via shared bus bars.  The resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    .. svgbob::

                    "capacitor"
         o1 ──┬──||||||||||||||──┬── o2
              │                  │
              │  ─────────────── │
              │  ─────────────── │
              │  ─────────────── │
              │   "inductor"     │

    Similar structures are described in
    :cite:`kimThinfilmSuperconductingResonator2011,chenCompactInductorcapacitorResonators2023`.

    Args:
        fingers: Number of interdigital capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger and bus bar in µm.
        n_turns: Number of horizontal meander inductor runs.
        wire_width: Width of the inductor wire in µm.
        wire_gap: Gap between adjacent inductor runs in µm.
        bus_bar_spacing: Vertical spacing between the capacitor and inductor sections in µm.
        etch_layer: Optional layer for the etch region.
        etch_bbox_margin: Margin around the structure for the etch region in µm.
        cross_section: Cross-section specification for the CPW ports.

    Returns:
        Component: A gdsfactory component with the lumped-element resonator
            geometry and two ports ('o1' and 'o2').
    """
    if n_turns % 2 == 0:
        raise ValueError(
            "n_turns must be odd so that the meander path spans from the "
            "left bus bar to the right bus bar"
        )
    if wire_gap <= 0:
        raise ValueError(f"wire_gap must be positive, got {wire_gap}")
    if bus_bar_spacing <= 0:
        raise ValueError(
            "bus_bar_spacing must be positive to electrically isolate the "
            "last inductor run from the full-width bus bar sections"
        )

    cap_width_check = 2 * finger_thickness + finger_length + finger_gap
    short_length_check = cap_width_check - 4 * wire_width
    if short_length_check <= 0:
        raise ValueError(
            f"Meander run length would be non-positive ({short_length_check} µm). "
            "Increase finger_length/finger_gap/finger_thickness or decrease wire_width."
        )

    c = Component()
    layer = LAYER.M1_DRAW

    cap_width = 2 * finger_thickness + finger_length + finger_gap
    cap_height = fingers * finger_thickness + (fingers - 1) * finger_gap
    ind_height = n_turns * wire_width + max(0, n_turns - 1) * wire_gap
    total_internal_height = cap_height + bus_bar_spacing + ind_height
    cap_y0 = ind_height + bus_bar_spacing

    _draw_interdigital_fingers_left(
        c,
        layer,
        x_offset=finger_thickness,
        y_offset=cap_y0,
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
    )
    _draw_interdigital_fingers_right(
        c,
        layer,
        x_offset=finger_thickness,
        y_offset=cap_y0,
        cap_width=cap_width,
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
    )

    pitch = wire_width + wire_gap
    last_y0 = (n_turns - 1) * pitch

    # Meander runs are inset wire_width from each narrow bus bar so that only
    # run 0 (via left tab) and run n_turns-1 (via right tab) connect to the
    # bus bars.  The bus bars remain narrow (wire_width) until cap_y0, ensuring
    # the full-width sections never share an edge with any inductor run.
    inner_x0 = 2 * wire_width
    short_length = cap_width - 4 * wire_width

    for i in range(n_turns):
        y0 = i * pitch
        c.add_polygon(
            [
                (inner_x0, y0),
                (inner_x0 + short_length, y0),
                (inner_x0 + short_length, y0 + wire_width),
                (inner_x0, y0 + wire_width),
            ],
            layer=layer,
        )

    for i in range(n_turns - 1):
        y0 = i * pitch + wire_width
        y1 = (i + 1) * pitch
        if i % 2 == 0:
            x0 = inner_x0 + short_length - wire_width
            c.add_polygon(
                [(x0, y0), (x0 + wire_width, y0), (x0 + wire_width, y1), (x0, y1)],
                layer=layer,
            )
        else:
            c.add_polygon(
                [
                    (inner_x0, y0),
                    (inner_x0 + wire_width, y0),
                    (inner_x0 + wire_width, y1),
                    (inner_x0, y1),
                ],
                layer=layer,
            )

    # Left tab: connects narrow left bus bar to run 0
    c.add_polygon(
        [
            (wire_width, 0),
            (inner_x0, 0),
            (inner_x0, wire_width),
            (wire_width, wire_width),
        ],
        layer=layer,
    )

    # Right tab: connects last run to narrow right bus bar
    c.add_polygon(
        [
            (inner_x0 + short_length, last_y0),
            (cap_width - wire_width, last_y0),
            (cap_width - wire_width, last_y0 + wire_width),
            (inner_x0 + short_length, last_y0 + wire_width),
        ],
        layer=layer,
    )

    # Left bus bar: narrow to cap_y0, then full width
    c.add_polygon(
        [(0, 0), (wire_width, 0), (wire_width, cap_y0), (0, cap_y0)],
        layer=layer,
    )
    c.add_polygon(
        [
            (0, cap_y0),
            (finger_thickness, cap_y0),
            (finger_thickness, total_internal_height),
            (0, total_internal_height),
        ],
        layer=layer,
    )

    # Right bus bar: starts at last_y0, narrow to cap_y0, then full width
    c.add_polygon(
        [
            (cap_width - wire_width, last_y0),
            (cap_width, last_y0),
            (cap_width, cap_y0),
            (cap_width - wire_width, cap_y0),
        ],
        layer=layer,
    )
    x_right = cap_width - finger_thickness
    c.add_polygon(
        [
            (x_right, cap_y0),
            (cap_width, cap_y0),
            (cap_width, total_internal_height),
            (x_right, total_internal_height),
        ],
        layer=layer,
    )

    if etch_layer is not None:
        c.add_polygon(
            [
                (-etch_bbox_margin, -etch_bbox_margin),
                (cap_width + etch_bbox_margin, -etch_bbox_margin),
                (
                    cap_width + etch_bbox_margin,
                    total_internal_height + etch_bbox_margin,
                ),
                (-etch_bbox_margin, total_internal_height + etch_bbox_margin),
            ],
            layer=etch_layer,
        )

    straight_cross_section = gf.get_cross_section(cross_section)
    straight_out = straight(
        length=etch_bbox_margin, cross_section=straight_cross_section
    )

    center_y = total_internal_height / 2
    straight_left = c.add_ref(straight_out).move((-etch_bbox_margin, center_y))
    straight_right = c.add_ref(straight_out).move((cap_width, center_y))

    c_additive = gf.boolean(
        A=c,
        B=c,
        operation="or",
        layer=layer,
        layer1=layer,
        layer2=straight_cross_section.layer,
    )

    c = gf.Component()
    c.absorb(c << c_additive)

    if etch_layer is not None:
        c_negative = gf.boolean(
            A=c,
            B=c_additive,
            operation="A-B",
            layer=etch_layer,
            layer1=etch_layer,
            layer2=layer,
        )
        c = gf.Component()
        c.absorb(c << c_additive)
        c.absorb(c << c_negative)

    c.add_port(
        name="o1",
        width=straight_left["o1"].width,
        center=straight_left["o1"].center,
        orientation=straight_left["o1"].orientation,
        layer=LAYER.M1_DRAW,
        port_type="electrical",
    )
    c.add_port(
        name="o2",
        width=straight_right["o2"].width,
        center=straight_right["o2"].center,
        orientation=straight_right["o2"].orientation,
        layer=LAYER.M1_DRAW,
        port_type="electrical",
    )

    c.move((-cap_width / 2, -total_internal_height / 2))

    total_wire_length = (
        2 * wire_width
        + n_turns * (cap_width - 4 * wire_width)
        + max(0, n_turns - 1) * wire_gap
    )
    c.info["total_wire_length"] = total_wire_length
    c.info["inductor_n_squares"] = total_wire_length / wire_width
    c.info["capacitor_fingers"] = fingers
    c.info["capacitor_finger_length"] = finger_length

    return c


def _draw_interdigital_fingers_left(
    c: Component,
    layer: LayerSpec,
    x_offset: float,
    y_offset: float,
    fingers: int,
    finger_length: float,
    finger_gap: float,
    thickness: float,
) -> None:
    """Draw left-side interdigital capacitor fingers (even-indexed, extending right)."""
    for i in range(ceil(fingers / 2)):
        finger_idx = 2 * i
        y0 = y_offset + finger_idx * (thickness + finger_gap)
        c.add_polygon(
            [
                (x_offset, y0),
                (x_offset + finger_length, y0),
                (x_offset + finger_length, y0 + thickness),
                (x_offset, y0 + thickness),
            ],
            layer=layer,
        )


def _draw_interdigital_fingers_right(
    c: Component,
    layer: LayerSpec,
    x_offset: float,
    y_offset: float,
    cap_width: float,
    fingers: int,
    finger_length: float,
    finger_gap: float,
    thickness: float,
) -> None:
    """Draw right-side interdigital capacitor fingers (odd-indexed, extending left)."""
    x_right_inner = cap_width - x_offset
    for i in range(floor(fingers / 2)):
        finger_idx = 1 + 2 * i
        y0 = y_offset + finger_idx * (thickness + finger_gap)
        c.add_polygon(
            [
                (x_right_inner - finger_length, y0),
                (x_right_inner, y0),
                (x_right_inner, y0 + thickness),
                (x_right_inner - finger_length, y0 + thickness),
            ],
            layer=layer,
        )


if __name__ == "__main__":
    from qpdk.helper import show_components

    show_components(
        meander_inductor,
        lumped_element_resonator,
    )
