"""Two transmons coupled via a flux-tunable coupler."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec

from qpdk.cells.junction import tunable_coupler_flux
from qpdk.cells.transmon import double_pad_transmon
from qpdk.tech import LAYER


@gf.cell_with_module_name
def transmons_with_tunable_coupler(
    transmon_spec: ComponentSpec = double_pad_transmon,
    coupler_spec: ComponentSpec = tunable_coupler_flux,
    transmon_separation: float = 300.0,
    coupler_offset_y: float = 0.0,
) -> Component:
    """Creates two transmon qubits coupled via a flux-tunable coupler.

    This example demonstrates a basic two-qubit system where the coupling
    strength can be dynamically controlled via magnetic flux bias of the
    tunable coupler.

    Args:
        transmon_spec: Component specification for the transmon qubits.
        coupler_spec: Component specification for the tunable coupler.
        transmon_separation: Center-to-center separation between transmons in μm.
        coupler_offset_y: Vertical offset of coupler relative to transmons in μm.

    Returns:
        Component: A gdsfactory component with two transmons and tunable coupler.
    """
    c = Component()

    # Create the two transmon qubits
    transmon1_ref = c << gf.get_component(transmon_spec)
    transmon2_ref = c << gf.get_component(transmon_spec)

    # Position transmons
    transmon1_ref.move((-transmon_separation / 2, 0))
    transmon2_ref.move((transmon_separation / 2, 0))

    # Create and position the tunable coupler
    coupler_ref = c << gf.get_component(coupler_spec)
    coupler_ref.move((0, coupler_offset_y))

    # Add metadata about the two-qubit system
    c.info["system_type"] = "two_qubit_tunable"
    c.info["transmon_separation"] = transmon_separation
    c.info["coupler_offset_y"] = coupler_offset_y

    # Add ports from individual components
    # Check if ports exist before adding them
    if "left_pad" in [p.name for p in transmon1_ref.ports]:
        c.add_port(name="transmon1_left", port=transmon1_ref.ports["left_pad"])
    if "right_pad" in [p.name for p in transmon1_ref.ports]:
        c.add_port(name="transmon1_right", port=transmon1_ref.ports["right_pad"])
    if "left_pad" in [p.name for p in transmon2_ref.ports]:
        c.add_port(name="transmon2_left", port=transmon2_ref.ports["left_pad"])
    if "right_pad" in [p.name for p in transmon2_ref.ports]:
        c.add_port(name="transmon2_right", port=transmon2_ref.ports["right_pad"])
    
    # Add coupler ports
    c.add_port(name="coupler_flux_bias", port=coupler_ref.ports["flux_bias"])
    c.add_port(name="coupler_left", port=coupler_ref.ports["left"])
    c.add_port(name="coupler_right", port=coupler_ref.ports["right"])

    return c


if __name__ == "__main__":
    from qpdk.helper import show_components
    show_components(transmons_with_tunable_coupler)