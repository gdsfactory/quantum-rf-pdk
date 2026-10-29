"""Indium bump components for 3D integration."""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec

from qpdk.cells._schematic import bump_transition_schematic, indium_bump_schematic
from qpdk.cells.tsv import _add_tapered_pad
from qpdk.tech import LAYER


@gf.cell(
    tags=["interconnects", "flip-chip", "3d-integration"],
    schematic_function=indium_bump_schematic,
)
def indium_bump(diameter: float = 15.0) -> Component:
    """Creates an indium bump component for 3D integration.

    See :cite:`rosenberg3DIntegratedSuperconducting2017` for details.

    Args:
        diameter: Diameter of the indium bump in µm.

    Returns:
        A gdsfactory Component representing the indium bump.
    """
    c = Component()
    circle = gf.components.circle(radius=diameter / 2, layer=LAYER.IND)
    ref = c.add_ref(circle)
    ref.move((0.0, 0.0))
    c.add_port(
        name="center",
        center=(
            0,
            0,
        ),
        orientation=0,
        layer=LAYER.IND,
        width=diameter,
        port_type="placement",
    )
    return c


@gf.cell(
    tags=["interconnects", "flip-chip", "3d-integration"],
    schematic_function=indium_bump_schematic,
)
def indium_bump_array(
    columns: int = 4,
    rows: int = 4,
    pitch: float = 200.0,
    diameter: float = 15.0,
) -> Component:
    """Creates a rectangular field of indium bumps centred at the origin.

    Use it to tie the ground planes of the two chips of a flip-chip stack
    together, away from any CPW gaps.

    .. svgbob::

        (o)  (o)  (o)  (o)
        (o)  (o)  (o)  (o)   <- rows x columns bumps at pitch
        (o)  (o)  (o)  (o)

    Args:
        columns: Number of bumps along x.
        rows: Number of bumps along y.
        pitch: Centre-to-centre pitch of the bumps along both axes in µm.
        diameter: Diameter of each indium bump in µm.

    Returns:
        A gdsfactory Component with ``columns * rows`` indium bumps.
    """
    c = Component()
    c.add_ref(
        indium_bump(diameter=diameter),
        columns=columns,
        rows=rows,
        column_pitch=pitch,
        row_pitch=pitch,
    ).move((-(columns - 1) * pitch / 2, -(rows - 1) * pitch / 2))
    return c


@gf.cell(
    tags=["interconnects", "flip-chip", "3d-integration", "waveguides"],
    schematic_function=bump_transition_schematic,
)
def bump_transition(
    cross_section: CrossSectionSpec = "cpw",
    cross_section_top: CrossSectionSpec = "cpw_top",
    taper_length: float = 100.0,
    taper_length_top: float = 100.0,
    pad_width: float = 40.0,
    pad_gap: float = 24.0,
    pad_length: float = 30.0,
    bump_diameter: float = 15.0,
    n_signal_bumps: int = 2,
    signal_bump_pitch: float = 20.0,
    n_ground_bumps_per_side: int = 4,
    ground_bump_pitch: float = 30.0,
    ground_bump_distance: float = 15.0,
) -> Component:
    r"""CPW transition between the two chips of a flip-chip stack through indium bumps.

    A bottom-chip CPW (``o1``, on ``M1``) tapers into a pad, the signal crosses
    the flip-chip gap through ``n_signal_bumps`` parallel indium bumps, and a
    top-chip CPW (``o2``, on ``M2``) tapers out of an identical pad in the
    opposite direction. ``2 * n_ground_bumps_per_side`` bumps tie the ground
    planes of the two chips together around the pad. This is the flip-chip
    counterpart of :func:`~qpdk.cells.tsv.tsv_transition_double_sided`, with
    bumps in place of vias, see
    :cite:`rosenberg3DIntegratedSuperconducting2017`.

    Both chips are drawn in the same (bottom-chip, top-down) coordinate frame,
    so the pads overlap and the bumps land on both.

    .. svgbob::

         bottom (M1)              top (M2)
               ______________________
        o1 ── /      (o)            \ ── o2
              \      (o)            /
               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
              (o)  (o)  (o)  (o)    <- ground bumps

    Args:
        cross_section: Bottom-chip CPW cross-section at ``o1``.
        cross_section_top: Top-chip CPW cross-section at ``o2``.
        taper_length: Length of the bottom-chip taper in µm.
        taper_length_top: Length of the top-chip taper in µm.
        pad_width: Centre-conductor width of both pads in µm.
        pad_gap: Gap between the pads and ground in µm.
        pad_length: Length of the pads along the propagation direction in µm.
        bump_diameter: Indium bump diameter in µm.
        n_signal_bumps: Number of parallel signal bumps in the pad.
        signal_bump_pitch: Centre-to-centre pitch of the signal bumps (along y)
            in µm.
        n_ground_bumps_per_side: Number of ground bumps on each side of the pads.
        ground_bump_pitch: Centre-to-centre pitch of the ground bumps (along x)
            in µm.
        ground_bump_distance: Distance from the ground-plane edge to the ground
            bump centres in µm.

    Returns:
        Component with the bottom-chip port ``o1`` and the top-chip port ``o2``.

    Raises:
        ValueError: If the signal bumps do not fit in the pad, or the ground
            bumps reach into the CPW gap.
    """
    signal_span = (n_signal_bumps - 1) * signal_bump_pitch + bump_diameter
    if signal_span > pad_width or bump_diameter > pad_length:
        msg = (
            f"{n_signal_bumps} signal bump(s) of {bump_diameter} µm at pitch "
            f"{signal_bump_pitch} µm do not fit in a {pad_length}x{pad_width} µm pad."
        )
        raise ValueError(msg)
    if ground_bump_distance < bump_diameter / 2:
        msg = (
            f"ground_bump_distance={ground_bump_distance} µm puts the ground bumps "
            "into the CPW gap."
        )
        raise ValueError(msg)

    c = Component()
    port_bottom = _add_tapered_pad(
        c, cross_section, pad_width, pad_gap, pad_length, taper_length, mirror=False
    )
    port_top = _add_tapered_pad(
        c,
        cross_section_top,
        pad_width,
        pad_gap,
        pad_length,
        taper_length_top,
        mirror=True,
    )

    bump = indium_bump(diameter=bump_diameter)
    for y in (np.arange(n_signal_bumps) - (n_signal_bumps - 1) / 2) * signal_bump_pitch:
        (c << bump).move((0, float(y)))
    y_ground = pad_width / 2 + pad_gap + ground_bump_distance
    xs = (
        np.arange(n_ground_bumps_per_side) - (n_ground_bumps_per_side - 1) / 2
    ) * ground_bump_pitch
    for x in xs:
        for y in (y_ground, -y_ground):
            (c << bump).move((float(x), y))

    c.add_port(name="o1", port=port_bottom)
    c.add_port(name="o2", port=port_top)
    return c
