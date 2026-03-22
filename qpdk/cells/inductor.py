"""Inductor and lumped-element resonator components."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER


@gf.cell
def meander_inductor(
    n_turns: int = 5,
    turn_length: float = 200.0,
    wire_width: float = 2.0,
    wire_gap: float = 4.0,
    etch_layer: LayerSpec | None = "M1_ETCH",
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
        wire_width: Width of the meander wire in µm.
        wire_gap: Gap between adjacent horizontal runs in µm.
        etch_layer: Optional layer for the etch bounding box around the inductor.
        etch_bbox_margin: Margin around the inductor for the etch bounding box in µm.

    Returns:
        Component: A gdsfactory component with the meander inductor geometry
            and two ports ('o1' and 'o2').
    """
    if n_turns < 1:
        raise ValueError("Must have at least 1 turn")
    if turn_length <= 0:
        raise ValueError(f"turn_length must be positive, got {turn_length}")
    if wire_width <= 0:
        raise ValueError(f"wire_width must be positive, got {wire_width}")
    if wire_gap <= 0:
        raise ValueError(f"wire_gap must be positive, got {wire_gap}")

    c = Component()
    layer = LAYER.M1_DRAW
    pitch = wire_width + wire_gap
    total_height = n_turns * wire_width + max(0, n_turns - 1) * wire_gap

    # Draw horizontal runs
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

    # Draw vertical connections between adjacent runs
    for i in range(n_turns - 1):
        y0 = i * pitch + wire_width
        y1 = (i + 1) * pitch
        if i % 2 == 0:
            # Connection at right end
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
            # Connection at left end
            c.add_polygon(
                [(0, y0), (wire_width, y0), (wire_width, y1), (0, y1)],
                layer=layer,
            )

    # Add etch bounding box
    if etch_layer is not None:
        c.add_polygon(
            [
                (-etch_bbox_margin, -etch_bbox_margin),
                (turn_length + etch_bbox_margin, -etch_bbox_margin),
                (turn_length + etch_bbox_margin, total_height + etch_bbox_margin),
                (-etch_bbox_margin, total_height + etch_bbox_margin),
            ],
            layer=etch_layer,
        )

    # Port o1: left side of the first (bottom) run
    c.add_port(
        name="o1",
        center=(0, wire_width / 2),
        width=wire_width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )

    # Port o2: depends on parity of n_turns
    last_run_center_y = (n_turns - 1) * pitch + wire_width / 2
    if n_turns % 2 == 1:
        # Odd: signal exits at right side of last run
        c.add_port(
            name="o2",
            center=(turn_length, last_run_center_y),
            width=wire_width,
            orientation=0,
            layer=layer,
            port_type="electrical",
        )
    else:
        # Even: signal exits at left side of last run
        c.add_port(
            name="o2",
            center=(0, last_run_center_y),
            width=wire_width,
            orientation=180,
            layer=layer,
            port_type="electrical",
        )

    # Center the component at the origin
    c.move((-turn_length / 2, -total_height / 2))

    # Store metadata
    total_wire_length = n_turns * turn_length + max(0, n_turns - 1) * wire_gap
    c.info["total_wire_length"] = total_wire_length
    c.info["n_squares"] = total_wire_length / wire_width

    return c


@gf.cell
def lumped_element_resonator(
    fingers: int = 20,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    finger_thickness: float = 5.0,
    n_turns: int = 5,
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
    c = Component()
    layer = LAYER.M1_DRAW

    # --- Capacitor section dimensions ---
    cap_width = 2 * finger_thickness + finger_length + finger_gap
    cap_height = fingers * finger_thickness + (fingers - 1) * finger_gap

    # --- Inductor section dimensions ---
    # The inductor turn length spans the internal width between bus bars
    inductor_turn_length = cap_width - 2 * finger_thickness
    ind_height = n_turns * wire_width + max(0, n_turns - 1) * wire_gap

    # --- Layout positions ---
    # Capacitor occupies the top section, inductor the bottom
    # Total internal height = cap + spacing + inductor
    total_internal_height = cap_height + bus_bar_spacing + ind_height

    # Capacitor top-left corner (before centering)
    cap_y0 = ind_height + bus_bar_spacing

    # Draw capacitor fingers (reuse interdigital_capacitor polygon logic)
    # Left-side fingers (connected to left bus bar)
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

    # Right-side fingers (connected to right bus bar)
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

    # --- Draw meander inductor section ---
    pitch = wire_width + wire_gap
    # The meander runs are centered between the bus bars
    meander_x0 = finger_thickness

    for i in range(n_turns):
        y0 = i * pitch
        c.add_polygon(
            [
                (meander_x0, y0),
                (meander_x0 + inductor_turn_length, y0),
                (meander_x0 + inductor_turn_length, y0 + wire_width),
                (meander_x0, y0 + wire_width),
            ],
            layer=layer,
        )

    # Vertical connections between meander runs
    for i in range(n_turns - 1):
        y0 = i * pitch + wire_width
        y1 = (i + 1) * pitch
        if i % 2 == 0:
            # Connection at right end
            x0 = meander_x0 + inductor_turn_length - wire_width
            c.add_polygon(
                [
                    (x0, y0),
                    (x0 + wire_width, y0),
                    (x0 + wire_width, y1),
                    (x0, y1),
                ],
                layer=layer,
            )
        else:
            # Connection at left end
            c.add_polygon(
                [
                    (meander_x0, y0),
                    (meander_x0 + wire_width, y0),
                    (meander_x0 + wire_width, y1),
                    (meander_x0, y1),
                ],
                layer=layer,
            )

    # --- Draw bus bars ---
    # Left bus bar: from bottom of inductor to top of capacitor
    c.add_polygon(
        [
            (0, 0),
            (finger_thickness, 0),
            (finger_thickness, total_internal_height),
            (0, total_internal_height),
        ],
        layer=layer,
    )

    # Right bus bar
    x_right = cap_width - finger_thickness
    c.add_polygon(
        [
            (x_right, 0),
            (cap_width, 0),
            (cap_width, total_internal_height),
            (x_right, total_internal_height),
        ],
        layer=layer,
    )

    # --- Add etch bounding box ---
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

    # --- Add CPW transitions for ports ---
    straight_cross_section = gf.get_cross_section(cross_section)
    straight_out = straight(
        length=etch_bbox_margin, cross_section=straight_cross_section
    )

    # Left port: at the center height of the structure
    center_y = total_internal_height / 2
    straight_left = c.add_ref(straight_out).move((-etch_bbox_margin, center_y))

    # Right port
    straight_right = c.add_ref(straight_out).move((cap_width, center_y))

    # --- Boolean operations to merge metal layers ---
    c_additive = gf.boolean(
        A=c,
        B=c,
        operation="or",
        layer=layer,
        layer1=layer,
        layer2=straight_cross_section.layer,
    )

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

    # Add ports
    c.add_port(
        name="o1",
        width=straight_left["o1"].width,
        center=straight_left["o1"].center,
        orientation=straight_left["o1"].orientation,
        layer=LAYER.M1_DRAW,
    )
    c.add_port(
        name="o2",
        width=straight_right["o2"].width,
        center=straight_right["o2"].center,
        orientation=straight_right["o2"].orientation,
        layer=LAYER.M1_DRAW,
    )

    # Center at origin
    c.move((-cap_width / 2, -total_internal_height / 2))

    # Store metadata
    total_wire_length = n_turns * inductor_turn_length + max(0, n_turns - 1) * wire_gap
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
    """Draw left-side interdigital capacitor fingers.

    Fingers extend to the right from the left bus bar.
    Even-indexed fingers (0, 2, 4, ...) belong to the left side.
    """
    from math import ceil

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
    """Draw right-side interdigital capacitor fingers.

    Fingers extend to the left from the right bus bar.
    Odd-indexed fingers (1, 3, 5, ...) belong to the right side.
    """
    from math import floor

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
