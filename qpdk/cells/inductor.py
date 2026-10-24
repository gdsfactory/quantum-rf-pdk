"""Inductor and lumped-element resonator components."""

from __future__ import annotations

from math import ceil, floor, sqrt

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells._schematic import (
    attach_schematic,
    lumped_element_resonator_schematic,
    meander_inductor_schematic,
)
from qpdk.cells.waveguides import straight
from qpdk.tech import (
    MEANDER_SQUARES_CONVENTIONS,
    MeanderCornerStyle,
    get_etch_section,
    get_meander_wire_gap,
    meander_inductor_cross_section,
    meander_n_squares_drawn,
    meander_round_radius,
    optimal_hairpin_arm_length,
)


@gf.cell(tags=["inductors"], schematic_function=meander_inductor_schematic)
def meander_inductor(
    n_turns: int = 5,
    turn_length: float = 200.0,
    cross_section: CrossSectionSpec = meander_inductor_cross_section,
    wire_gap: float | None = None,
    etch_bbox_margin: float = 2.0,
    add_etch: bool = True,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
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
            specification. The meander's vertical pitch is the wire width
            plus the run-to-run gap, which defaults to twice the etch gap
            so that the etched regions of adjacent runs do not overlap,
            maintaining the characteristic impedance of each run.
        wire_gap: Optional explicit gap between adjacent inductor runs in µm.
            If None (default), it's inferred as 2x the etch gap from the
            cross-section, or the wire width if it has no etch section.
        etch_bbox_margin: Extra margin around the inductor for the etch bounding box in µm.
            This margin is added in addition to the etch region defined in the cross-section.
        add_etch: Whether to add the etch bounding box. Defaults to True.
        corner_style: Shape of the U-turn corners, all within the same bounding box:

            * ``"sharp"``: Manhattan corners (the original geometry).
            * ``"chamfer"``: the outer corner of each 90° bend is cut at 45° so
              the wire keeps its width :math:`w` along the diagonal.
            * ``"round"``: circular arcs of centreline radius ``radius``.
            * ``"optimal"``: the current-crowding-free hairpin contour of
              :cite:`clemGeometrydependentCriticalCurrents2011`, drawn with
              :func:`gdsfactory.components.optimal_hairpin`. Each hairpin spans
              half a run.
        radius: Centreline radius of ``"round"`` corners in µm, with
            :math:`w/2 < R \\le p/2` for pitch :math:`p`. Defaults to :math:`p/2`,
            which makes each U-turn a semicircle. Ignored for other styles.

    Returns:
        Component: A gdsfactory component with the meander inductor geometry
            and two ports ('o1' and 'o2'). ``info`` holds

            * ``total_wire_length``: centreline length of the sharp meander in µm.
            * ``n_squares``: ``total_wire_length / w``, which counts every corner
              square as one square. The SAX model uses this value.
            * ``n_squares_drawn``: squares of the drawn polygon for the chosen
              ``corner_style``, from :func:`qpdk.tech.meander_n_squares_drawn`.
              A sharp 90° corner counts 0.559 squares
              :cite:`hagedornRightAngleBendsThin1963`.
            * ``n_squares_convention``: the corner rule used for
              ``n_squares_drawn``.

    Raises:
        ValueError: If `n_turns` < 1, `turn_length` <= 0, `wire_gap` is
            not positive, `radius` is out of range, the runs are too short for
            the chosen corners, or `corner_style` is unknown.
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

    # Same rule as the SAX model: explicit gap wins, else 2x etch width,
    # else the wire width when there is no etch section
    wire_gap = get_meander_wire_gap(xs, wire_gap)

    c = Component()
    pitch = wire_width + wire_gap
    total_height = n_turns * wire_width + max(0, n_turns - 1) * wire_gap

    match corner_style:
        case "sharp":
            _draw_sharp_meander(c, n_turns, turn_length, wire_width, pitch, layer)
        case "chamfer":
            _draw_chamfer_meander(c, n_turns, turn_length, wire_width, pitch, layer)
        case "round":
            r = meander_round_radius(wire_width, wire_gap, radius)
            _draw_round_meander(c, n_turns, turn_length, wire_width, pitch, r, layer)
        case "optimal":
            _draw_optimal_meander(c, n_turns, turn_length, wire_width, pitch, layer)
        case _:
            raise ValueError(f"Unknown corner_style {corner_style!r}")
    if corner_style != "sharp":
        c.flatten()

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
    c.info["n_squares_drawn"] = meander_n_squares_drawn(
        n_turns, turn_length, wire_width, wire_gap, corner_style, radius
    )
    c.info["n_squares_convention"] = MEANDER_SQUARES_CONVENTIONS[corner_style]
    c.info["cross_section_name"] = xs.name

    return c


def _draw_sharp_meander(
    c: Component,
    n_turns: int,
    turn_length: float,
    wire_width: float,
    pitch: float,
    layer: LayerSpec,
) -> None:
    """Draws Manhattan runs and vertical connectors, run 0 at y = 0."""
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


def _draw_chamfer_meander(
    c: Component,
    n_turns: int,
    turn_length: float,
    wire_width: float,
    pitch: float,
    layer: LayerSpec,
) -> None:
    r"""Draws the sharp meander with each outer bend corner cut at 45°.

    The cut triangle has legs :math:`(2 - \sqrt{2})\,w`, so the wire width
    across the diagonal stays :math:`w`.
    """
    cut = (2 - sqrt(2)) * wire_width
    length, w = turn_length, wire_width
    for i in range(n_turns):
        y0 = i * pitch
        y1 = y0 + w
        bend_right_below = i % 2 == 0 and i < n_turns - 1
        bend_right_above = i % 2 == 1
        bend_left_below = i % 2 == 1 and i < n_turns - 1
        bend_left_above = i % 2 == 0 and i >= 2
        pts = [(0.0, y0)] if not bend_left_below else [(cut, y0)]
        if bend_right_below:
            pts += [(length - cut, y0), (length, y0 + cut)]
        else:
            pts += [(length, y0)]
        if bend_right_above:
            pts += [(length, y1 - cut), (length - cut, y1)]
        else:
            pts += [(length, y1)]
        if bend_left_above:
            pts += [(cut, y1), (0.0, y1 - cut)]
        else:
            pts += [(0.0, y1)]
        if bend_left_below:
            pts += [(0.0, y0 + cut)]
        c.add_polygon(pts, layer=layer)

    for i in range(n_turns - 1):
        y0 = i * pitch + w
        y1 = (i + 1) * pitch
        x0 = length - w if i % 2 == 0 else 0.0
        c.add_polygon([(x0, y0), (x0 + w, y0), (x0 + w, y1), (x0, y1)], layer=layer)


def _draw_round_meander(
    c: Component,
    n_turns: int,
    turn_length: float,
    wire_width: float,
    pitch: float,
    radius: float,
    layer: LayerSpec,
) -> None:
    """Draws the meander as one extruded path with circular U-turns.

    The arcs are placed so that the outer wire edges stay on the bounding box of
    the sharp meander.

    Raises:
        ValueError: If the runs are too short for the radius.
    """
    w, r = wire_width, radius
    first_last = turn_length - w / 2 - r
    middle = turn_length - w - 2 * r
    vertical = pitch - 2 * r
    if n_turns > 1 and (first_last < 0 or (n_turns > 2 and middle < 0)):
        minimum = w + 2 * r if n_turns > 2 else w / 2 + r
        raise ValueError(
            f"turn_length={turn_length} is too short for round corners of "
            f"radius {r}; need at least {minimum}"
        )

    def _straight(path: gf.Path, length: float) -> gf.Path:
        return path + gf.path.straight(length=length) if length > 0 else path

    path = gf.Path([(0, 0)])
    if n_turns == 1:
        path = _straight(path, turn_length)
    else:
        for i in range(n_turns):
            run = first_last if i in {0, n_turns - 1} else middle
            path = _straight(path, run)
            if i == n_turns - 1:
                break
            angle = 90 if i % 2 == 0 else -90
            path += gf.path.arc(radius=r, angle=angle)
            path = _straight(path, vertical)
            path += gf.path.arc(radius=r, angle=angle)
    wire = gf.path.extrude(path, layer=layer, width=w)
    ref = c << wire
    ref.move((0, w / 2))


def _draw_optimal_meander(
    c: Component,
    n_turns: int,
    turn_length: float,
    wire_width: float,
    pitch: float,
    layer: LayerSpec,
) -> None:
    """Draws the meander from chained Clem-Berggren hairpins.

    Each U-turn is one :func:`gdsfactory.components.optimal_hairpin` spanning
    half a run. Adjacent hairpins meet at the middle of the shared run.
    """
    w = wire_width
    if n_turns == 1:
        c.add_polygon([(0, 0), (turn_length, 0), (turn_length, w), (0, w)], layer=layer)
        return

    # Validates that the hairpin fits; raises ValueError otherwise.
    optimal_hairpin_arm_length(turn_length, w, pitch)
    hairpin = gf.c.optimal_hairpin(
        width=w,
        pitch=pitch,
        length=turn_length / 2,
        turn_ratio=(pitch + w) / w,
        layer=layer,
    )
    refs = []
    for i in range(n_turns - 1):
        ref = c << hairpin
        if i == 0:
            ref.move(ref.ports["e2"].center, (turn_length / 2, w / 2))
        elif i % 2 == 1:
            ref.connect("e1", refs[-1].ports["e1"])
        else:
            ref.connect("e2", refs[-1].ports["e2"])
        refs.append(ref)

    half = turn_length / 2
    c.add_polygon([(0, 0), (half, 0), (half, w), (0, w)], layer=layer)
    y0 = (n_turns - 1) * pitch
    x0 = 0.0 if n_turns % 2 == 0 else half
    c.add_polygon(
        [(x0, y0), (x0 + half, y0), (x0 + half, y0 + w), (x0, y0 + w)], layer=layer
    )


attach_schematic(meander_inductor, meander_inductor_schematic)


@gf.cell(
    tags=["resonators", "inductors", "capacitors"],
    schematic_function=lumped_element_resonator_schematic,
)
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

        f_\text{r} = \frac{1}{2\pi\sqrt{LC}}

    .. svgbob::

       +-----------+
       | Capacitor |
       o1 --+   (IDC)   +-- o2
       |           |
       | Inductor  |
       | (Meander) |
       +-----------+

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
        port_type="optical",
        cross_section=xs,
    )
    c.add_port(
        name="o2",
        port=straight_right.ports["o2"],
        layer=layer,
        port_type="optical",
        cross_section=xs,
    )

    c.info["total_wire_length"] = (
        2 * wire_width + n_turns * short_length + max(0, n_turns - 1) * wire_gap
    )
    c.info["inductor_n_squares"] = c.info["total_wire_length"] / wire_width
    c.info["capacitor_fingers"] = fingers
    c.info["capacitor_finger_length"] = finger_length

    return c


attach_schematic(lumped_element_resonator, lumped_element_resonator_schematic)


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
