r"""Gatemon qubit components.

A gatemon is a transmon whose Josephson junction is a gate-tunable weak link
made from a quantum conductor, such as a semiconducting nanowire, graphene or a
carbon nanotube. A gate voltage :math:`V_\text{g}` tunes the junction's
critical current and so the Josephson energy :math:`E_\text{J}`, and thereby
the qubit frequency, without a flux bias.

The layout follows the grounded single-island design of
:cite:`riechertCarbonNanotubeGatemon2025`: a charge island in a pocket etched
from the ground plane forms the shunt capacitor, the junction connects it to
the ground plane, and a DC gate line ends next to the junction.

References:
    - :cite:`riechertCarbonNanotubeGatemon2025`
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from kfactory.conf import CheckInstances

from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER
from qpdk.utils import add_rect, subtract_draw_from_etch

__all__ = ["gatemon"]


@gf.cell(check_instances=CheckInstances.IGNORE, tags=["qubits"])
def gatemon(
    island_size: tuple[float, float] = (150.0, 150.0),
    pocket_gap: float = 60.0,
    pocket_gap_top: float = 40.0,
    electrode_size: tuple[float, float] = (40.0, 10.0),
    electrode_spacing: float = 60.0,
    lead_width: float = 2.0,
    junction_length: float = 1.0,
    junction_width: float = 0.2,
    size_patch: tuple[float, float] = (1.5, 1.0),
    gate_offset: float = 40.0,
    gate_tip_width: float = 2.0,
    gate_tip_gap: float = 5.0,
    gate_length: float = 50.0,
    gate_cross_section: CrossSectionSpec = "cpw",
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    r"""Creates a grounded gatemon qubit with a gate-tunable junction and a DC gate line.

    A single charge island sits in a pocket etched from the ground plane and
    implements the shunt capacitor that sets :math:`E_\text{C}`. Below the
    island, two junction electrodes are joined by a thin lead containing the
    weak link. The upper electrode connects to the island and the lower
    electrode to the ground plane at the bottom edge of the pocket. A gate
    line enters through the bottom of the pocket as a coplanar line and ends
    in a narrow tip next to the junction. A readout resonator couples to the
    island across the top gap of the pocket.

    .. svgbob::

        +-------------------------------+
        |           readout             |  pocket
        |      +-----------------+      |  (M1_ETCH)
        |      |                 |      |
        |      |     island      |      |
        |      |                 |      |
        |      +-------+---------+      |
        |          +---+---+            |
        |          +---+---+  electrode |
        |              |                |
        |              X  <---+         |
        |              |      |  gate   |
        |          +---+---+  |         |
        +----------+-------+--+  +------+
          ground plane        |  |
                              |  |
                              gate

    Note:
        Only the M1 metal and the junction marker layers are drawn. The
        junction weak link (e.g. a nanotube) is marked on ``JJ_AREA`` and
        ``JJ_PATCH``, and the gate electrode is drawn in M1. The top-gate
        dielectric and electrode stack (e.g. hBN and a top gate) of
        :cite:`riechertCarbonNanotubeGatemon2025` is not drawn, since QPDK
        has no layers for it.

    Args:
        island_size: (width, height) of the charge island in µm.
        pocket_gap: Etched gap between the island and the left and right walls
            of the pocket in µm.
        pocket_gap_top: Etched gap between the island and the top edge of the
            pocket, where the readout couples, in µm.
        electrode_size: (width, height) of each junction electrode in µm.
        electrode_spacing: Distance between the inner edges of the two
            junction electrodes in µm.
        lead_width: Width of the lead between the junction electrodes in µm.
        junction_length: Length of the weak link, i.e. the gap in the lead
            bridged by the junction, in µm.
        junction_width: Width of the junction marker on ``JJ_AREA`` in µm.
        size_patch: (width, height) of the ``JJ_PATCH`` contact patches at the
            two ends of the weak link in µm.
        gate_offset: Horizontal distance from the lead to the centre line of
            the gate line in µm.
        gate_tip_width: Width of the gate tip pointing at the junction in µm.
        gate_tip_gap: Gap between the gate tip and the lead in µm.
        gate_length: Length of the gate line beyond the bottom edge of the
            pocket in µm.
        gate_cross_section: Cross-section of the gate line where it leaves
            the pocket.
        layer_metal: Layer for the metal.
        layer_etch: Layer for the etched pocket.

    Returns:
        Component: A gdsfactory component with the gatemon geometry.

    Raises:
        ValueError: If the gate line would overlap the junction electrodes.
    """
    c = Component()

    island_width, island_height = island_size
    electrode_width, electrode_height = electrode_size
    xs_gate = gf.get_cross_section(gate_cross_section)

    if gate_offset - xs_gate.width / 2 <= electrode_width / 2:
        raise ValueError(
            f"gate_offset={gate_offset} µm places the gate line on the junction "
            f"electrodes; it must exceed {electrode_width / 2 + xs_gate.width / 2} µm"
        )

    # Vertical stack below the island, from top to bottom
    upper_electrode_y1 = -island_height / 2
    upper_electrode_y0 = upper_electrode_y1 - electrode_height
    lower_electrode_y1 = upper_electrode_y0 - electrode_spacing
    pocket_bottom = lower_electrode_y1 - electrode_height
    junction_y = (upper_electrode_y0 + lower_electrode_y1) / 2

    # Charge island
    add_rect(
        c,
        layer=layer_metal,
        x_center=0,
        width=island_width,
        y_center=0,
        height=island_height,
    )

    # Junction electrodes; the upper one touches the island, the lower one the ground
    for y0, y1 in (
        (upper_electrode_y0, upper_electrode_y1),
        (pocket_bottom, lower_electrode_y1),
    ):
        add_rect(c, layer=layer_metal, x_center=0, width=electrode_width, y0=y0, y1=y1)

    # Lead between the electrodes, interrupted by the weak link
    add_rect(
        c,
        layer=layer_metal,
        x_center=0,
        width=lead_width,
        y0=junction_y + junction_length / 2,
        y1=upper_electrode_y0,
    )
    add_rect(
        c,
        layer=layer_metal,
        x_center=0,
        width=lead_width,
        y0=lower_electrode_y1,
        y1=junction_y - junction_length / 2,
    )

    # Junction markers: weak link on JJ_AREA, contact patches on JJ_PATCH
    junction_marker_height = junction_length + size_patch[1]
    add_rect(
        c,
        layer=LAYER.JJ_AREA,
        x_center=0,
        width=junction_width,
        y_center=junction_y,
        height=junction_marker_height,
    )
    for sign in (1, -1):
        add_rect(
            c,
            layer=LAYER.JJ_PATCH,
            x_center=0,
            width=size_patch[0],
            y_center=junction_y + sign * junction_length / 2,
            height=size_patch[1],
        )

    # Gate line inside the pocket, ending in a tip next to the junction
    gate_x0 = gate_offset - xs_gate.width / 2
    add_rect(
        c,
        layer=layer_metal,
        x_center=gate_offset,
        width=xs_gate.width,
        y0=pocket_bottom,
        y1=junction_y + gate_tip_width / 2,
    )
    add_rect(
        c,
        layer=layer_metal,
        x0=lead_width / 2 + gate_tip_gap,
        x1=gate_x0,
        y_center=junction_y,
        height=gate_tip_width,
    )

    # Pocket etched from the ground plane, keeping the drawn metal
    pocket = gf.Component()
    add_rect(
        pocket,
        layer=layer_etch,
        x_center=0,
        width=island_width + 2 * pocket_gap,
        y0=pocket_bottom,
        y1=island_height / 2 + pocket_gap_top,
    )
    subtract_draw_from_etch(
        component=c,
        etch_shape=pocket,
        etch_layer=layer_etch,
        draw_layer=layer_metal,
    )

    # Gate line leaving the pocket through the ground plane
    gate_ref = c.add_ref(straight(length=gate_length, cross_section=xs_gate))
    gate_ref.rotate(-90)
    gate_ref.move((
        gate_offset - gate_ref.ports["o1"].x,
        pocket_bottom - gate_ref.ports["o1"].y,
    ))
    gate_port = gate_ref.ports["o2"].copy()
    c.absorb(gate_ref)

    c.add_port(
        name="readout",
        center=(0, island_height / 2),
        width=island_width,
        orientation=90,
        layer=layer_metal,
    )
    c.add_port(name="gate", port=gate_port)
    c.add_port(
        name="junction",
        center=(0, junction_y),
        width=gf.snap.snap_to_grid(junction_marker_height, grid_factor=2),
        orientation=90,
        layer=LAYER.JJ_AREA,
        port_type="placement",
    )

    c.info["qubit_type"] = "gatemon"

    return c
