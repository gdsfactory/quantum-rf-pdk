"""Physical sections of a capacitively coupled quarter-wave resonator."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from qpdk.cells._schematic import resonator_meander_schematic
from qpdk.cells.waveguides import bend_circular

__all__ = ["resonator_meander"]


@gf.cell(tags=["resonators"], schematic_function=resonator_meander_schematic)
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
    component.info["length"] = meander_ref.cell.info["length"]
    component.flatten()
    return component
