"""Resonator components."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells.waveguides import bend_circular, straight
from qpdk.cells.capacitor import interdigital_capacitor


@gf.cell_with_module_name
def resonator(
    length: float = 4000.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    *,
    open_start: bool = False,
    open_end: bool = False,
) -> Component:
    """Creates a meandering coplanar waveguide resonator.

    Changing `open_start` and `open_end` appropriately allows creating
    a shorted quarter-wave resonator or an open half-wave resonator.

    See :cite:`m.pozarMicrowaveEngineering2012` for details

    Args:
        length: Length of the resonator in μm.
        meanders: Number of meander sections to fit the resonator in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the resonator.
        open_start: If True, adds an etch section at the start of the resonator.
        open_end: If True, adds an etch section at the end of the resonator.

    Returns:
        Component: A gdsfactory component with meandering resonator geometry.
    """
    c = Component()
    cross_section = gf.get_cross_section(cross_section)
    bend = gf.get_component(bend_spec, cross_section=cross_section, angle=180)
    length_per_one_straight = (length - meanders * bend.info["length"]) / (meanders + 1)

    if length_per_one_straight <= 0:
        raise ValueError(
            f"Resonator length {length} is too short for {meanders} meanders with current bend spec {bend}. "
            f"Increase length, reduce meanders, or change the bend spec."
        )

    straight_comp = straight(
        length=length_per_one_straight,
        cross_section=cross_section,
    )

    # Route meandering quarter-wave resonator
    previous_port = None
    for i in range(meanders):
        straight_ref = c.add_ref(straight_comp)
        bend_ref = c.add_ref(bend)

        if i == 0:
            first_straight_ref = straight_ref
        else:  # i > 0
            straight_ref.connect("o1", previous_port)

        if i % 2 == 0:
            bend_ref.mirror()
            bend_ref.rotate(90)

        bend_ref.connect("o1", straight_ref.ports["o2"])
        previous_port = bend_ref.ports["o2"]

    # Final straight section
    final_straight = straight(
        length=length_per_one_straight,
        cross_section=cross_section,
    )
    final_straight_ref = c.add_ref(final_straight)
    final_straight_ref.connect("o1", previous_port)

    # Etch at the open end
    if open_end or open_start:
        cross_section_etch_section = next(
            s
            for s in gf.get_cross_section(cross_section).sections
            if "etch_offset" in s.name
        )

        open_etch_comp = gf.c.rectangle(
            size=(
                cross_section_etch_section.width,
                2 * cross_section_etch_section.width + cross_section.width,
            ),
            layer=cross_section_etch_section.layer,
            centered=True,
            port_type="optical",
            port_orientations=(0, 180),
        )

        def _add_etch_at_port(port_name, ref_port, output_port):
            """Helper function to add etch at a specific port."""
            open_etch = c.add_ref(open_etch_comp)
            open_etch.connect(
                port_name,
                ref_port,
                allow_width_mismatch=True,
                allow_layer_mismatch=True,
            )
            c.add_port(output_port, port=open_etch.ports[output_port])

        if open_end:
            _add_etch_at_port("o1", final_straight_ref.ports["o2"], "o2")
        if open_start:
            _add_etch_at_port("o2", first_straight_ref.ports["o1"], "o1")

    if not open_end:
        c.add_port("o2", port=final_straight_ref.ports["o2"])

    if not open_start:
        c.add_port("o1", port=first_straight_ref.ports["o1"])

    # Add metadata
    c.info["length"] = length
    c.info["resonator_type"] = "quarter_wave"
    # c.info["frequency_estimate"] = (
    #     3e8 / (4 * length * 1e-6) / 1e9
    # )  # GHz, rough estimate

    return c


# A quarter-wave resonator is shorted at one end and has maximum electric field
# at the open end, making it suitable for capacitive coupling.
resonator_quarter_wave = partial(resonator, open_start=False, open_end=True)
# A half-wave resonator is open at both ends
resonator_half_wave = partial(resonator, open_start=True, open_end=True)


@gf.cell_with_module_name
def inductor(
    length: float = 2000.0,
    meanders: int = 4,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "microstrip_narrow",
    width: float = 50.0,
) -> Component:
    """Creates a meandering inductor using narrow metal lines.

    An inductor component using Manhattan routing with narrow wires to provide
    inductance for lumped-element resonators. The inductance scales approximately
    with the total length and inversely with the wire width.

    See :cite:`kimThinfilmSuperconductingResonator2011` for lumped-element resonator design.

    Args:
        length: Total length of the inductor wire in μm.
        meanders: Number of meander sections to fit the inductor in a compact area.
        bend_spec: Specification for the bend component used in meanders.
        cross_section: Cross-section specification for the inductor wire.
        width: Total width constraint for the inductor layout in μm.

    Returns:
        Component: A gdsfactory component with meandering inductor geometry.
    """
    c = Component()
    cross_section = gf.get_cross_section(cross_section)
    bend = gf.get_component(bend_spec, cross_section=cross_section, angle=180)
    
    # Calculate how much space each meander takes
    bend_length = bend.info.get("length", 0)
    total_bend_length = meanders * bend_length
    
    if length <= total_bend_length:
        raise ValueError(
            f"Inductor length {length} is too short for {meanders} meanders. "
            f"Minimum length needed: {total_bend_length:.2f} μm"
        )
    
    # Length available for straight sections
    straight_length_total = length - total_bend_length
    length_per_straight = straight_length_total / (meanders + 1)
    
    # Create the meander pattern
    straight_comp = straight(
        length=length_per_straight,
        cross_section=cross_section,
    )
    
    # Start with first straight section
    first_straight_ref = c.add_ref(straight_comp)
    previous_port = first_straight_ref.ports["o2"]
    
    # Add meanders
    for i in range(meanders):
        # Add bend
        bend_ref = c.add_ref(bend)
        
        # Alternate bend orientation for compact layout
        if i % 2 == 0:
            bend_ref.mirror()
            bend_ref.rotate(90)
        
        bend_ref.connect("o1", previous_port)
        
        # Add next straight section
        if i < meanders:  # Not the last iteration
            next_straight_ref = c.add_ref(straight_comp)
            next_straight_ref.connect("o1", bend_ref.ports["o2"])
            previous_port = next_straight_ref.ports["o2"]
        else:
            previous_port = bend_ref.ports["o2"]
    
    # Add final straight section
    final_straight_ref = c.add_ref(straight_comp)
    final_straight_ref.connect("o1", previous_port)
    
    # Add ports
    c.add_port("o1", port=first_straight_ref.ports["o1"])
    c.add_port("o2", port=final_straight_ref.ports["o2"])
    
    # Add metadata
    c.info["length"] = length
    c.info["meanders"] = meanders
    c.info["inductor_type"] = "meandering"
    
    return c


@gf.cell_with_module_name
def lumped_resonator(
    inductor_length: float = 1500.0,
    inductor_meanders: int = 3,
    capacitor_fingers: int = 6,
    capacitor_finger_length: float = 30.0,
    capacitor_finger_gap: float = 2.0,
    capacitor_thickness: float = 5.0,
    spacing: float = 50.0,
    cross_section_inductor: CrossSectionSpec = "microstrip_narrow",
) -> Component:
    """Creates a lumped-element resonator combining an inductor and capacitor.

    A lumped-element resonator consists of discrete inductance and capacitance
    elements that determine the resonant frequency. This design provides better
    control over the LC values compared to distributed resonators.

    See :cite:`kimThinfilmSuperconductingResonator2011` for design principles.

    Args:
        inductor_length: Total length of the inductor wire in μm.
        inductor_meanders: Number of meander sections in the inductor.
        capacitor_fingers: Number of fingers in the interdigital capacitor.
        capacitor_finger_length: Length of each capacitor finger in μm.
        capacitor_finger_gap: Gap between capacitor fingers in μm.
        capacitor_thickness: Thickness of capacitor fingers in μm.
        spacing: Spacing between inductor and capacitor in μm.
        cross_section_inductor: Cross-section for the inductor wire.

    Returns:
        Component: A gdsfactory component with the lumped resonator geometry.
    """
    c = Component()
    
    # Create inductor
    inductor_comp = inductor(
        length=inductor_length,
        meanders=inductor_meanders,
        cross_section=cross_section_inductor,
    )
    inductor_ref = c.add_ref(inductor_comp)
    
    # Create capacitor
    capacitor_comp = interdigital_capacitor(
        fingers=capacitor_fingers,
        finger_length=capacitor_finger_length,
        finger_gap=capacitor_finger_gap,
        thickness=capacitor_thickness,
    )
    capacitor_ref = c.add_ref(capacitor_comp)
    
    # Position capacitor relative to inductor
    # Place capacitor to the right of the inductor with some spacing
    inductor_bbox = inductor_ref.bbox()
    capacitor_ref.move((inductor_bbox.right + spacing, 0))
    
    # Connect inductor and capacitor with a short connecting wire
    connection_length = spacing
    connection = straight(
        length=connection_length,
        cross_section=cross_section_inductor,
    )
    connection_ref = c.add_ref(connection)
    connection_ref.connect("o1", inductor_ref.ports["o2"])
    
    # The connection should connect to the capacitor
    # Note: This is a simplified connection - in practice, you might need
    # more sophisticated routing depending on the exact layout
    
    # Add ports for external connections
    c.add_port("inductor_in", port=inductor_ref.ports["o1"])
    c.add_port("capacitor_out", port=capacitor_ref.ports["o2"])
    
    # Add metadata
    c.info["resonator_type"] = "lumped_element"
    c.info["inductor_length"] = inductor_length
    c.info["capacitor_fingers"] = capacitor_fingers
    c.info["frequency_estimate_note"] = "f = 1/(2*π*√(LC))"
    
    return c

if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()
    c = resonator_quarter_wave()
    # c = resonator()
    # c = resonator(open_start=True, open_end=True)
    c.show()
