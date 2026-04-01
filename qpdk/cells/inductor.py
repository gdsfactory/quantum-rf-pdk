"""Inductor and lumped-element resonator components."""

from __future__ import annotations

from math import ceil, floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
from qpdk.tech import (
    get_etch_section,
    meander_inductor_cross_section,
)


@gf.cell(tags=("inductors",))
def meander_inductor(
    n_turns: int = 5,
    turn_length: float = 200.0,
    cross_section: CrossSectionSpec = meander_inductor_cross_section,
    wire_gap: float | None = None,
    etch_bbox_margin: float = 2.0,
    add_etch: bool = True,
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
        wire_gap: Optional explicit gap between adjacent inductor runs in µm.
            If None (default), it's inferred as 2x the etch gap from the cross-section.
        etch_bbox_margin: Extra margin around the inductor for the etch bounding box in µm.
            This margin is added in addition to the etch region defined in the cross-section.
        add_etch: Whether to add the etch bounding box. Defaults to True.

    Returns:
        Component: A gdsfactory component with the meander inductor geometry
            and two ports ('o1' and 'o2').

    Raises:
        ValueError: If `n_turns` < 1 or `turn_length` <= 0.
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
        etch_section = get_etch_section(xs)
        etch_layer = etch_section.layer
    except ValueError:
        etch_section = None
        etch_layer = None

    # For CPW-like structures, we assume a pitch that allows for non-overlapping etches
    # i.e. pitch = width + 2 * gap, which means wire_gap = 2 * etch_width
    # If no etch section is found, we use a default gap equal to the wire width
    if wire_gap is None:
        wire_gap = 2 * etch_section.width if etch_section is not None else wire_width

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

    if add_etch and etch_section is not None:
        # Extra margin on top of the implicit etch margin from the cross-section
        margin = etch_section.width + etch_bbox_margin
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
            cross_section=xs,
        )
    else:
        c.add_port(
            name="o2",
            center=(0, last_run_center_y),
            width=wire_width,
            orientation=180,
            layer=layer,
            cross_section=xs,
        )

    c.move((-turn_length / 2, -total_height / 2))

    total_wire_length = n_turns * turn_length + max(0, n_turns - 1) * wire_gap
    c.info["total_wire_length"] = total_wire_length
    c.info["n_squares"] = total_wire_length / wire_width
    c.info["cross_section"] = xs.name

    return c


@gf.cell(tags=("resonators", "inductors", "capacitors"))
def lumped_element_resonator(
    fingers: int = 20,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    finger_thickness: float = 5.0,
    n_turns: int = 15,
    bus_bar_spacing: float = 4.0,
    cross_section: CrossSectionSpec = meander_inductor_cross_section,
    etch_bbox_margin: float = 2.0,
) -> Component:
    r"""Creates a lumped-element resonator combining an interdigital capacitor and a meander inductor.

    The resonator consists of an interdigital capacitor section (providing
    capacitance) connected in parallel with a meander inductor section
    (providing inductance) via shared bus bars.  The resonance frequency is:

    .. math::

        f_r = \frac{1}{2\pi\sqrt{LC}}

    .. svgbob::

              ┌──────────────────┐
              │ ┌──┐ ┌──┐ ┌──┐  │
              │ └──┘ └──┘ └──┘  │
              │  ┌──┐ ┌──┐ ┌──┐│
              │  └──┘ └──┘ └──┘│
              │  "capacitor"    │
         o1 ──┤  (interdigital) ├── o2
              │                 │
              │ ────────────────│
              │─────────────────│
              │ ────────────────│
              │─────────────────│
              │ ────────────────│
              │  "inductor"     │
              │  (meander)      │
              └─────────────────┘

    Similar structures are described in
    :cite:`kimThinfilmSuperconductingResonator2011,chenCompactInductorcapacitorResonators2023`.

    Args:
        fingers: Number of interdigital capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger and bus bar in µm.
        n_turns: Number of horizontal meander inductor runs.
        bus_bar_spacing: Vertical spacing between the capacitor and inductor sections in µm.
        cross_section: Cross-section specification for the inductor and ports.
        etch_bbox_margin: Margin around the structure for the etch region in µm.

    Returns:
        Component: A gdsfactory component with the lumped-element resonator
            geometry and two ports ('o1' and 'o2').

    Raises:
        ValueError: If `n_turns` is even, `bus_bar_spacing` <= 0, or if the
            resultant meander run length is non-positive.
    """
    if n_turns % 2 == 0:
        raise ValueError(
            "n_turns must be odd so that the meander path spans from the "
            "left bus bar to the right bus bar"
        )
    if bus_bar_spacing <= 0:
        raise ValueError(
            "bus_bar_spacing must be positive to electrically isolate the "
            "last inductor run from the full-width bus bar sections"
        )

    xs = gf.get_cross_section(cross_section)
    wire_width = xs.width
    etch_section = get_etch_section(xs)
    wire_gap = 2 * etch_section.width
    layer = xs.layer
    etch_layer = etch_section.layer
    etch_width = etch_section.width

    cap_width = 2 * finger_thickness + finger_length + finger_gap
    short_length = cap_width - 4 * wire_width
    if short_length <= 0:
        raise ValueError(
            f"Meander run length would be non-positive ({short_length} µm). "
            "Increase finger_length/finger_gap/finger_thickness or decrease wire_width."
        )

    c = Component()

    # 1. Inductor part
    ind = c << meander_inductor(
        n_turns=n_turns,
        turn_length=short_length,
        cross_section=cross_section,
        etch_bbox_margin=0,
    )

    cap_height = fingers * finger_thickness + (fingers - 1) * finger_gap
    ind_height = ind.size_info.height
    total_internal_height = cap_height + bus_bar_spacing + ind_height

    # Center inductor at the bottom of the internal area
    ind.dcenter = (0, -total_internal_height / 2 + ind_height / 2)

    # 2. Capacitor part (fingers and bus bars)
    cap_y0 = -total_internal_height / 2 + ind_height + bus_bar_spacing

    x_left_inner = -cap_width / 2 + finger_thickness
    x_right_inner = cap_width / 2 - finger_thickness

    _draw_interdigital_fingers_left(
        c,
        layer,
        x_inner=x_left_inner,
        y_offset=cap_y0,
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
    )
    _draw_interdigital_fingers_right(
        c,
        layer,
        x_inner=x_right_inner,
        y_offset=cap_y0,
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=finger_thickness,
    )

    # 3. Bus bars connecting everything
    # Small overlap to ensure solid connectivity
    overlap = 0.1

    # Left bus bar: connects to turn 0 (bottom)
    # Use the metal bottom edge of the inductor, not the component bbox bottom (which includes etch)
    left_bb_ymin = ind.ports["o1"].center[1] - wire_width / 2
    c.add_polygon(
        [
            (-cap_width / 2, left_bb_ymin),
            (-cap_width / 2 + wire_width, left_bb_ymin),
            (-cap_width / 2 + wire_width, cap_y0 + overlap),
            (-cap_width / 2, cap_y0 + overlap),
        ],
        layer=layer,
    )
    # Top wide part
    c.add_polygon(
        [
            (-cap_width / 2, cap_y0),
            (-cap_width / 2 + finger_thickness, cap_y0),
            (-cap_width / 2 + finger_thickness, total_internal_height / 2),
            (-cap_width / 2, total_internal_height / 2),
        ],
        layer=layer,
    )

    # Right bus bar: connects to turn n_turns-1 (top)
    # Redundant section below top run is removed
    right_bb_ymin = ind.ports["o2"].center[1] - wire_width / 2
    c.add_polygon(
        [
            (cap_width / 2 - wire_width, right_bb_ymin),
            (cap_width / 2, right_bb_ymin),
            (cap_width / 2, cap_y0 + overlap),
            (cap_width / 2 - wire_width, cap_y0 + overlap),
        ],
        layer=layer,
    )
    # Top wide part
    c.add_polygon(
        [
            (cap_width / 2 - finger_thickness, cap_y0),
            (cap_width / 2, cap_y0),
            (cap_width / 2, total_internal_height / 2),
            (cap_width / 2 - finger_thickness, total_internal_height / 2),
        ],
        layer=layer,
    )

    # Tabs to inductor
    # Left tab connects o1 to the left bus bar
    c.add_polygon(
        [
            (
                -cap_width / 2 + wire_width - overlap,
                ind.ports["o1"].center[1] - wire_width / 2,
            ),
            (
                ind.ports["o1"].center[0] + overlap,
                ind.ports["o1"].center[1] - wire_width / 2,
            ),
            (
                ind.ports["o1"].center[0] + overlap,
                ind.ports["o1"].center[1] + wire_width / 2,
            ),
            (
                -cap_width / 2 + wire_width - overlap,
                ind.ports["o1"].center[1] + wire_width / 2,
            ),
        ],
        layer=layer,
    )
    # Right tab connects o2 to the right bus bar
    c.add_polygon(
        [
            (
                ind.ports["o2"].center[0] - overlap,
                ind.ports["o2"].center[1] - wire_width / 2,
            ),
            (
                cap_width / 2 - wire_width + overlap,
                ind.ports["o2"].center[1] - wire_width / 2,
            ),
            (
                cap_width / 2 - wire_width + overlap,
                ind.ports["o2"].center[1] + wire_width / 2,
            ),
            (
                ind.ports["o2"].center[0] - overlap,
                ind.ports["o2"].center[1] + wire_width / 2,
            ),
        ],
        layer=layer,
    )

    # 4. Etch bounding box
    margin = etch_width + etch_bbox_margin
    c.add_polygon(
        [
            (-cap_width / 2 - margin, -total_internal_height / 2 - margin),
            (cap_width / 2 + margin, -total_internal_height / 2 - margin),
            (cap_width / 2 + margin, total_internal_height / 2 + margin),
            (-cap_width / 2 - margin, total_internal_height / 2 + margin),
        ],
        layer=etch_layer,
    )

    # 5. Ports
    straight_out = straight(length=margin, cross_section=cross_section)
    center_y = 0
    straight_left = c.add_ref(straight_out).move((-cap_width / 2 - margin, center_y))
    straight_right = c.add_ref(straight_out).move((cap_width / 2, center_y))

    c_metal = gf.boolean(
        A=c, B=c, operation="or", layer=layer, layer1=layer, layer2=xs.layer
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
        port=straight_left.ports["o1"],
        layer=layer,
        port_type="electrical",
        cross_section=xs,
    )
    c.add_port(
        name="o2",
        port=straight_right.ports["o2"],
        layer=layer,
        port_type="electrical",
        cross_section=xs,
    )

    c.info["total_wire_length"] = (
        2 * wire_width + n_turns * short_length + max(0, n_turns - 1) * wire_gap
    )
    c.info["inductor_n_squares"] = c.info["total_wire_length"] / wire_width
    c.info["capacitor_fingers"] = fingers
    c.info["capacitor_finger_length"] = finger_length

    return c


def _draw_interdigital_fingers_left(
    c: Component,
    layer: LayerSpec,
    x_inner: float,
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
                (x_inner, y0),
                (x_inner + finger_length, y0),
                (x_inner + finger_length, y0 + thickness),
                (x_inner, y0 + thickness),
            ],
            layer=layer,
        )


def _draw_interdigital_fingers_right(
    c: Component,
    layer: LayerSpec,
    x_inner: float,
    y_offset: float,
    fingers: int,
    finger_length: float,
    finger_gap: float,
    thickness: float,
) -> None:
    """Draw right-side interdigital capacitor fingers (odd-indexed, extending left)."""
    for i in range(floor(fingers / 2)):
        finger_idx = 1 + 2 * i
        y0 = y_offset + finger_idx * (thickness + finger_gap)
        c.add_polygon(
            [
                (x_inner - finger_length, y0),
                (x_inner, y0),
                (x_inner, y0 + thickness),
                (x_inner - finger_length, y0 + thickness),
            ],
            layer=layer,
        )


if __name__ == "__main__":
    from qpdk.helper import show_components

    show_components(
        meander_inductor,
        lumped_element_resonator,
    )
