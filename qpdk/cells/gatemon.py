r"""Nanotube gatemon layout following Fig. 1b of Riechert et al. (2025).

A charge island in an etched ground-plane pocket couples to a readout resonator.
Two Nb/Au contacts connect the island to a transferred nanotube weak link, and an
Al top gate controls it through a local hBN flake.

References:
    - :cite:`riechertCarbonNanotubeGatemon2025`
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec
from kfactory.conf import CheckInstances

from qpdk.cells._schematic import (
    attach_schematic,
    gatemon_coupler_schematic,
    gatemon_schematic,
)
from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER
from qpdk.utils import add_rect, subtract_draw_from_etch

__all__ = ["gatemon", "gatemon_coupler"]


@gf.cell(
    check_instances=CheckInstances.IGNORE,
    tags=["qubits"],
    schematic_function=gatemon_schematic,
)
def gatemon(
    island_size: tuple[float, float] = (150.0, 150.0),
    pocket_gap: float = 60.0,
    pocket_gap_top: float = 40.0,
    electrode_size: tuple[float, float] = (60.0, 60.0),
    electrode_spacing: float = 100.0,
    lead_width: float = 2.0,
    junction_length: float = 1.0,
    junction_width: float = 0.002,
    junction_offset: float = 20.0,
    nanotube_contact_overlap: float = 0.5,
    hbn_size: tuple[float, float] = (80.0, 80.0),
    size_patch: tuple[float, float] = (60.0, 50.0),
    electrode_offset: float = -40.0,
    gate_offset: float = 100.0,
    gate_tip_width: float = 16.0,
    gate_tip_gap: float = 5.0,
    gate_length: float = 50.0,
    gate_cross_section: CrossSectionSpec = "cpw",
    layer_metal: LayerSpec = LAYER.M1_DRAW,
    layer_etch: LayerSpec = LAYER.M1_ETCH,
) -> Component:
    r"""Create a gatemon with contact pads, rectangular hBN and a boxy top gate.

    The upper Nb/Au pad overlaps the island and its lower counterpart overlaps
    ground. Thin contact stems approach each other beneath the island's left
    half, then branch right to the nanotube. The top gate has a rectangular tip
    over the weak link and a right-angle elbow onto exposed M1.
    A rectangular hBN footprint encloses the tube and insulates every
    gate/contact overlap. Al reconnection pads overlap the Nb/Au and M1 pads.

    The contact proportions follow Fig. 1b of
    :cite:`riechertCarbonNanotubeGatemon2025`, with idealized rectangular hBN
    and gate geometry. Register the actual flake and nanotube positions before
    fabrication. ``CNT`` is a native GDS placement path of nominal width, with
    overlap onto both contacts. The weak link belongs in the circuit model and
    is excluded from the PEC extrusions in ``qpdk.tech.LAYER_STACK_GATEMON``.

    Args:
        island_size: Island (width, height) in µm.
        pocket_gap: Gap between the island and the pocket side walls in µm.
        pocket_gap_top: Gap from the island to the readout side of the pocket in µm.
        electrode_size: Nb/Au landing-pad (width, height) in µm.
        electrode_spacing: Gap between the two landing pads in µm.
        lead_width: Width of the Nb/Au stems and contact branches in µm.
        junction_length: Gap between the contact tips in µm.
        junction_width: Width of the nanotube placement trace in µm.
        junction_offset: Horizontal distance from the stems to the weak link in µm.
        nanotube_contact_overlap: Tube overlap onto each contact tip in µm.
        hbn_size: Rectangular hBN footprint (width, height) in µm.
        size_patch: Al contact-reconnection pad (width, height) in µm.
        electrode_offset: Horizontal position of the contact stems relative to the island.
        gate_offset: Horizontal distance from the stems to the M1 gate feed in µm.
        gate_tip_width: Width of the top-gate tip and neck in µm.
        gate_tip_gap: Lateral clearance from hBN to the M1 gate landing in µm.
        gate_length: M1 gate-feed length beyond the pocket in µm.
        gate_cross_section: Cross-section of the gate feed at the cell boundary.
        layer_metal: Layer for M1 metal.
        layer_etch: Layer for the etched pocket.

    Returns:
        Gatemon with readout, gate and junction placement ports.

    Raises:
        ValueError: If dimensions are invalid, pads or gate leave the pocket,
            the nanotube is exposed, or a gate/contact overlap lacks hBN.
    """
    c = Component()
    iw, ih = island_size
    ew, eh = electrode_size
    hw, hh = hbn_size
    xs = gf.get_cross_section(gate_cross_section)
    if (
        min(
            *island_size,
            pocket_gap,
            pocket_gap_top,
            *electrode_size,
            electrode_spacing,
            lead_width,
            junction_length,
            junction_width,
            junction_offset,
            nanotube_contact_overlap,
            *hbn_size,
            *size_patch,
            gate_tip_width,
            gate_tip_gap,
            gate_length,
        )
        <= 0
    ):
        raise ValueError("Gatemon dimensions must be positive")
    if junction_width < 2 * c.kcl.dbu:
        raise ValueError(
            "Nanotube placement width must span at least two database units"
        )
    if junction_width > lead_width or nanotube_contact_overlap > lead_width:
        raise ValueError("Nanotube trace and overlap must fit on the contact tips")
    if junction_length + 2 * lead_width >= electrode_spacing:
        raise ValueError("The contact branches must fit between the electrode pads")
    if lead_width > ew or size_patch[0] > ew or size_patch[1] > 0.85 * eh:
        raise ValueError(
            "Contact leads and reconnection patches must fit the electrodes"
        )
    if abs(electrode_offset) + ew / 2 > iw / 2:
        raise ValueError("The junction electrodes must lie under the island")

    pocket_half_width = iw / 2 + pocket_gap
    gate_x = electrode_offset + gate_offset
    junction_x = electrode_offset + junction_offset
    gate_half_extent = max(
        gate_tip_width / 2,
        *(abs(section.offset) + section.width / 2 for section in xs.sections),
    )
    if gate_offset - xs.width / 2 <= ew / 2:
        raise ValueError("The gate line overlaps the junction electrodes")
    if gate_x + gate_half_extent >= pocket_half_width:
        raise ValueError("The gate line is outside the pocket")

    # The optical image shows the upper pad mostly on the island and the lower pad on ground.
    upper_bottom = -ih / 2 - 0.15 * eh
    lower_top = upper_bottom - electrode_spacing
    pocket_bottom = lower_top - 0.2 * eh
    junction_y = (upper_bottom + lower_top) / 2
    stem_right = electrode_offset + lead_width / 2
    gate_tip_left = junction_x - gate_tip_width / 2
    if gate_tip_left <= stem_right:
        raise ValueError("The gate tip must clear the left contact stems")
    hbn_left = (stem_right + gate_tip_left) / 2
    gate_landing_x = hbn_left + hw + gate_tip_gap
    if gate_landing_x >= gate_x - xs.width / 2:
        raise ValueError(
            "The hBN footprint and gate landing must fit before the gate line"
        )
    if hbn_left <= -pocket_half_width:
        raise ValueError("The hBN footprint is outside the pocket")
    if junction_y - hh / 2 <= pocket_bottom or junction_y + hh / 2 >= -ih / 2:
        raise ValueError("The hBN footprint must fit between island and ground")

    add_rect(c, layer=layer_metal, x_center=0, width=iw, y_center=0, height=ih)
    for y0, y1 in ((upper_bottom, upper_bottom + eh), (lower_top - eh, lower_top)):
        add_rect(
            c,
            layer=LAYER.CNT_CONTACT,
            x_center=electrode_offset,
            width=ew,
            y0=y0,
            y1=y1,
        )
    for y0, y1 in (
        (junction_y + junction_length / 2, upper_bottom),
        (lower_top, junction_y - junction_length / 2),
    ):
        add_rect(
            c,
            layer=LAYER.CNT_CONTACT,
            x_center=electrode_offset,
            width=lead_width,
            y0=y0,
            y1=y1,
        )
    for sign in (-1, 1):
        add_rect(
            c,
            layer=LAYER.CNT_CONTACT,
            x0=electrode_offset - lead_width / 2,
            x1=junction_x + lead_width / 2,
            y_center=junction_y + sign * (junction_length + lead_width) / 2,
            height=lead_width,
        )

    nanotube_length = junction_length + 2 * nanotube_contact_overlap
    c.shapes(c.kcl.layout.layer(*LAYER.CNT)).insert(
        gf.kdb.DPath(
            [
                gf.kdb.DPoint(junction_x, junction_y - nanotube_length / 2),
                gf.kdb.DPoint(junction_x, junction_y + nanotube_length / 2),
            ],
            junction_width,
        )
    )
    add_rect(
        c,
        layer=LAYER.HBN,
        x0=hbn_left,
        x1=hbn_left + hw,
        y_center=junction_y,
        height=hh,
    )
    for y in (-ih / 2 + size_patch[1] / 2, lower_top - eh / 2):
        add_rect(
            c,
            layer=LAYER.JJ_PATCH,
            x_center=electrode_offset,
            width=size_patch[0],
            y_center=y,
            height=size_patch[1],
        )

    add_rect(
        c,
        layer=layer_metal,
        x_center=gate_x,
        width=xs.width,
        y0=pocket_bottom,
        y1=junction_y + gate_tip_width / 2,
    )
    add_rect(
        c,
        layer=layer_metal,
        x0=gate_landing_x,
        x1=gate_x + xs.width / 2,
        y_center=junction_y,
        height=gate_tip_width,
    )
    add_rect(
        c,
        layer=LAYER.TOP_GATE,
        x0=junction_x - gate_tip_width / 2,
        x1=gate_x + gate_tip_width / 2,
        y_center=junction_y,
        height=gate_tip_width,
    )
    add_rect(
        c,
        layer=LAYER.TOP_GATE,
        x_center=gate_x,
        width=gate_tip_width,
        y0=pocket_bottom + gate_tip_gap,
        y1=junction_y + gate_tip_width / 2,
    )

    hbn = c.get_region(LAYER.HBN)
    tube = c.get_region(LAYER.CNT)
    if not (tube - hbn).is_empty():
        raise ValueError("hBN must enclose the entire nanotube")
    if not (
        (c.get_region(LAYER.TOP_GATE) & c.get_region(LAYER.CNT_CONTACT)) - hbn
    ).is_empty():
        raise ValueError("hBN must insulate every top-gate contact overlap")
    if not (hbn & c.get_region(LAYER.JJ_PATCH)).is_empty():
        raise ValueError("hBN must leave the reconnection patches exposed")

    pocket = Component()
    add_rect(
        pocket,
        layer=layer_etch,
        x_center=0,
        width=2 * pocket_half_width,
        y0=pocket_bottom,
        y1=ih / 2 + pocket_gap_top,
    )
    subtract_draw_from_etch(
        component=c, etch_shape=pocket, etch_layer=layer_etch, draw_layer=layer_metal
    )
    gate_ref = c.add_ref(straight(length=gate_length, cross_section=xs))
    gate_ref.rotate(-90)
    gate_ref.move((
        gate_x - gate_ref.ports["o1"].x,
        pocket_bottom - gate_ref.ports["o1"].y,
    ))
    gate_port = gate_ref.ports["o2"].copy()
    c.absorb(gate_ref)
    c.add_port(
        name="readout",
        center=(0, ih / 2),
        width=iw,
        orientation=90,
        layer=layer_metal,
    )
    c.add_port(name="gate", port=gate_port)
    c.add_port(
        name="junction",
        center=(junction_x, junction_y),
        width=gf.snap.snap_to_grid(junction_width, grid_factor=2),
        orientation=90,
        layer=LAYER.CNT,
        port_type="placement",
    )
    c.info["qubit_type"] = "gatemon"
    return c


attach_schematic(gatemon, gatemon_schematic)


@gf.cell(tags=["couplers"], schematic_function=gatemon_coupler_schematic)
def gatemon_coupler(
    length: float = 600.0,
    coupling_position: float = 200.0,
    island_width: float = 150.0,
    pocket_gap_top: float = 40.0,
    qubit_side: str = "bottom",
    coupling_capacitance: float = 5.4e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> Component:
    """Return a passing CPW with a capacitive island reference port.

    The qubit port sits at the island edge across the etched gap. Its circuit
    branch represents a lumped capacitance, so no metal spur crosses that gap.
    The capacitance is a nominal circuit estimate, not a layout extraction.

    Args:
        length: Complete CPW length in µm.
        coupling_position: Island center measured from o1 in µm.
        island_width: Width of the island reference plane in µm.
        pocket_gap_top: Clearance from island to the CPW conductor edge in µm.
        qubit_side: Island side of the CPW, bottom or top.
        coupling_capacitance: Nominal mutual capacitance in F for the SAX model.
        cross_section: Cross-section of the passing CPW.

    Returns:
        CPW with o1, o2 and qubit ports.

    Raises:
        ValueError: If the island reference plane lies outside the line.
    """
    if (
        not 0 < coupling_position < length
        or min(island_width, pocket_gap_top) <= 0
        or coupling_capacitance < 0
        or qubit_side not in {"bottom", "top"}
    ):
        raise ValueError("Invalid gatemon coupling reference plane")
    c = Component()
    line = c << straight(length=length, cross_section=cross_section)
    c.add_ports(line.ports)
    xs = gf.get_cross_section(cross_section)
    side = -1 if qubit_side == "bottom" else 1
    gap = xs.width / 2 + pocket_gap_top
    c.add_port(
        "qubit",
        center=(coupling_position, side * gap),
        width=island_width,
        orientation=270 if side < 0 else 90,
        layer=xs.layer,
    )
    c.info["length"] = length
    c.info["coupling_capacitance"] = coupling_capacitance
    return c


attach_schematic(gatemon_coupler, gatemon_coupler_schematic)
