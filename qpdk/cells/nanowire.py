"""Lumped resonator made of a high-kinetic-inductance nanowire."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from qpdk.cells.inductor import meander_inductor
from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER, MeanderCornerStyle, get_etch_section
from qpdk.utils import add_rect

__all__ = ["nanowire_resonator"]

NANOWIRE_WIDTH_RANGE = (0.05, 1.0)
"""Allowed nanowire width range in µm."""


@gf.cell(tags=["resonators", "inductors"])
def nanowire_resonator(
    wire_width: float = 0.1,
    wire_length: float = 40.0,
    n_turns: int = 1,
    wire_gap: float | None = None,
    corner_style: MeanderCornerStyle = "sharp",
    radius: float | None = None,
    pad_width: float = 300.0,
    pad_height: float = 150.0,
    pad_ground_gap: float = 50.0,
    coupling_gap: float = 4.0,
    contact_overlap: float = 2.0,
    feedline_length: float = 500.0,
    cross_section: CrossSectionSpec = "cpw",
    wire_layer: LayerSpec = LAYER.NbTiN,
) -> Component:
    r"""Creates a lumped resonator from a high-kinetic-inductance nanowire.

    A nanowire on ``wire_layer`` joins two capacitor pads. The upper pad sits
    ``coupling_gap`` below the centre conductor of a CPW feedline and couples to
    it capacitively. The lower pad is tied to the ground plane along its bottom
    edge, so the resonator is a parallel LC to ground, shunt-coupled to the
    feedline:

    .. svgbob::

        o1 ────────────────────────────── o2   feedline
             ┌──────────────────────┐
             │       pad (C)        │
             └──────────┬───────────┘
                        │  nanowire (L)
             ┌──────────┴───────────┐
             │     ground pad       │
        ─────┘                      └─────   ground

    The wire is straight for ``n_turns=1``, or a :func:`meander_inductor` with
    vertical runs of length ``wire_length`` otherwise, so ``n_turns`` must be odd
    for the wire to end on the opposite pad. ``info["n_squares"]`` is the drawn
    square count between the pad edges from
    :func:`qpdk.tech.meander_n_squares_drawn`; the ``contact_overlap`` stubs under
    the pads are not counted.

    Design references for the nanowire geometry and its kinetic inductance are
    :cite:`joshiStrongKineticinductanceKerr2022,maleevaCircuitQuantumElectrodynamics2018`.
    The matching SAX model is :func:`qpdk.models.nanowire_resonator`.

    Args:
        wire_width: Nanowire width in µm, between 0.05 and 1.
        wire_length: Length of each nanowire run in µm, which is also the gap
            between the two pads.
        n_turns: Number of nanowire runs (odd). 1 gives a straight wire.
        wire_gap: Gap between adjacent runs in µm. Defaults to ``wire_width``.
        corner_style: Corner style of a meandered wire, see
            :func:`meander_inductor`.
        radius: Centreline radius for ``"round"`` corners in µm.
        pad_width: Pad size along the feedline in µm.
        pad_height: Pad size across the feedline in µm.
        pad_ground_gap: Gap between the pads and the ground plane at the sides
            in µm.
        coupling_gap: Gap between the feedline centre conductor and the upper pad
            in µm.
        contact_overlap: Length of the nanowire stubs that overlap the pads in µm.
        feedline_length: Length of the feedline in µm.
        cross_section: Cross-section of the feedline.
        wire_layer: Layer of the nanowire.

    Returns:
        Component with feedline ports ``o1`` and ``o2``.

    Raises:
        ValueError: If the wire width is out of range, ``n_turns`` is even or
            not positive, a length is not positive, or the wire does not fit
            between the pads or the feedline is too short.
    """
    w_min, w_max = NANOWIRE_WIDTH_RANGE
    if not w_min <= wire_width <= w_max:
        raise ValueError(
            f"wire_width must be between {w_min} and {w_max} µm, got {wire_width}"
        )
    if n_turns < 1 or n_turns % 2 == 0:
        raise ValueError(
            f"n_turns must be a positive odd number so the wire ends on the "
            f"opposite pad, got {n_turns}"
        )
    for name, value in {
        "wire_length": wire_length,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "pad_ground_gap": pad_ground_gap,
        "coupling_gap": coupling_gap,
        "contact_overlap": contact_overlap,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if contact_overlap > pad_height:
        raise ValueError(
            f"contact_overlap={contact_overlap} must not exceed pad_height={pad_height}"
        )

    xs = gf.get_cross_section(cross_section)
    metal_layer = xs.layer
    etch_layer = get_etch_section(xs).layer
    centre_width = xs.width

    window_width = pad_width + 2 * pad_ground_gap
    if feedline_length < window_width:
        raise ValueError(
            f"feedline_length={feedline_length} must be at least "
            f"pad_width + 2 * pad_ground_gap = {window_width}"
        )

    wire_xs = gf.cross_section.cross_section(width=wire_width, layer=wire_layer)
    wire_gap = wire_width if wire_gap is None else wire_gap
    wire = meander_inductor(
        n_turns=n_turns,
        turn_length=wire_length,
        cross_section=wire_xs,
        wire_gap=wire_gap,
        add_etch=False,
        corner_style=corner_style,
        radius=radius,
    )
    wire_extent = n_turns * wire_width + (n_turns - 1) * wire_gap
    if wire_extent > pad_width:
        raise ValueError(
            f"The meander is {wire_extent} µm wide, more than pad_width={pad_width}"
        )

    c = Component()

    feed = c << straight(length=feedline_length, cross_section=cross_section)
    feed.dcenter = (0, 0)

    pad_top = -centre_width / 2 - coupling_gap
    pad_bottom = pad_top - pad_height
    ground_pad_top = pad_bottom - wire_length
    ground_pad_bottom = ground_pad_top - pad_height

    add_rect(
        c,
        etch_layer,
        x_center=0,
        width=window_width,
        y0=ground_pad_bottom,
        y1=-centre_width / 2,
    )
    for y0 in (pad_bottom, ground_pad_bottom):
        add_rect(c, metal_layer, x_center=0, width=pad_width, y0=y0, y1=y0 + pad_height)

    wire_ref = c << wire
    wire_ref.rotate(90)
    wire_ref.dcenter = (0, pad_bottom - wire_length / 2)
    for port in wire_ref.ports:
        x, y = port.dcenter
        inward = 1 if y > pad_bottom - wire_length / 2 else -1
        add_rect(
            c,
            wire_layer,
            x_center=x,
            width=wire_width,
            y0=min(y, y + inward * contact_overlap),
            y1=max(y, y + inward * contact_overlap),
        )

    c.add_ports(feed.ports)
    c.info["n_squares"] = wire.info["n_squares_drawn"]
    c.info["n_squares_convention"] = wire.info["n_squares_convention"]
    c.info["wire_gap"] = wire_gap
    return c
