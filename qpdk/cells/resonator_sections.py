"""Physical sections of a capacitively coupled quarter-wave resonator."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells._schematic import (
    resonator_coupling_region_schematic,
    resonator_meander_schematic,
)
from qpdk.cells.waveguides import bend_circular, straight


@gf.cell(
    tags=("resonators", "couplers"),
    schematic_function=resonator_coupling_region_schematic,
)
def resonator_coupling_region(
    length: float = 300.0,
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
    cross_section_non_resonator: CrossSectionSpec = "cpw",
) -> Component:
    """Return the open-ended resonator segment beside its coupling bus.

    This is a modeled leaf in recursive circuit netlists. Its four ports are
    the bus ends, the open resonator end, and the meander attachment point.

    Args:
        length: Length of the resonator segment in µm.
        coupling_straight_length: Length of the adjacent bus in µm.
        coupling_gap: Gap between the bus and resonator in µm.
        cross_section: Resonator cross section.
        cross_section_non_resonator: Bus cross section.
    """
    component = Component()
    resonator_ref = component << gf.get_component(
        "resonator",
        length=length,
        meanders=0,
        cross_section=cross_section,
        open_start=True,
        open_end=False,
    )
    bus_cross_section = gf.get_cross_section(cross_section_non_resonator)
    bus_ref = component << straight(
        length=coupling_straight_length, cross_section=bus_cross_section
    )
    bus_ref.movey(coupling_gap + bus_cross_section.width)
    bus_ref.xmin = resonator_ref.ports["o1"].x

    component.add_port("resonator_o1", port=resonator_ref.ports["o1"])
    component.add_port("resonator_o2", port=resonator_ref.ports["o2"])
    component.add_port("coupling_o1", port=bus_ref.ports["o1"])
    component.add_port("coupling_o2", port=bus_ref.ports["o2"])
    component.info["length"] = length
    component.info["coupling_length"] = coupling_straight_length
    component.info["coupling_gap"] = coupling_gap
    component.flatten()
    return component


resonator_coupling_region.schematic_function = resonator_coupling_region_schematic


@gf.cell(tags=("resonators",), schematic_function=resonator_meander_schematic)
def resonator_meander(
    length: float = 3700.0,
    meanders: int = 6,
    bend_spec: ComponentSpec = bend_circular,
    cross_section: CrossSectionSpec = "cpw",
    end_with_bend: bool = False,
) -> Component:
    """Return the modeled CPW path from the first bend to the shorted end.

    Args:
        length: Total path length, including bends, in µm.
        meanders: Number of meander bends.
        bend_spec: Component used for each bend.
        cross_section: Resonator cross section.
        end_with_bend: Whether the path ends with a bend.
    """
    component = Component()
    meander_ref = component << gf.get_component(
        "resonator",
        length=length,
        meanders=meanders,
        bend_spec=bend_spec,
        cross_section=cross_section,
        start_with_bend=True,
        end_with_bend=end_with_bend,
        open_start=False,
        open_end=False,
    )
    component.add_ports(meander_ref.ports)
    component.info["length"] = length
    component.flatten()
    return component


resonator_meander.schematic_function = resonator_meander_schematic
