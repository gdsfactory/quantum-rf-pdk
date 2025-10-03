"""Airbridge components for superconducting quantum circuits.

This module provides airbridge components for crossing transmission lines
without electrical contact, essential for complex quantum circuit layouts.
Airbridges are elevated metal structures that span over underlying circuits.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

import qpdk.tech as tech
from qpdk.tech import LAYER


@gf.cell
def airbridge(
    bridge_length: float = 30.0,
    bridge_width: float = 8.0,
    pad_width: float = 15.0,
    pad_length: float = 12.0,
    bridge_layer: LayerSpec = LAYER.AB_DRAW,
    pad_layer: LayerSpec = LAYER.AB_VIA,
) -> Component:
    """Generate a superconducting airbridge component.

    Creates an airbridge consisting of a suspended bridge span and landing pads
    on either side. The bridge allows transmission lines to cross over each other
    without electrical contact, which is essential for complex quantum circuit
    routing without crosstalk.

    The bridge_layer (AB_DRAW) represents the elevated metal bridge structure,
    while the pad_layer (AB_VIA) represents the contact/landing pads that connect
    to the underlying circuit.

    Note:
        To be used with :class:`~gdsfactory.cross_section.ComponentAlongPath`
        the unrotated version should be *oriented for placement on a horizontal line*.

    Args:
        bridge_length: Total length of the airbridge span in µm.
        bridge_width: Width of the suspended bridge in µm.
        pad_width: Width of the landing pads in µm.
        pad_length: Length of each landing pad in µm.
        bridge_layer: Layer for the suspended bridge metal (default: AB_DRAW).
        pad_layer: Layer for the landing pads/contacts (default: AB_VIA).

    Returns:
        Component containing the airbridge geometry with appropriate ports.
    """
    c = gf.Component()

    # Create the suspended bridge
    c << gf.components.rectangle(
        size=(bridge_length, bridge_width),
        layer=bridge_layer,
        centered=True,
    )

    # Create left landing pad
    left_pad = c << gf.components.rectangle(
        size=(pad_length, pad_width),
        layer=pad_layer,
        centered=True,
    )
    left_pad.move((-bridge_length / 2 - pad_length / 2, 0))

    # Create right landing pad
    right_pad = c << gf.components.rectangle(
        size=(pad_length, pad_width),
        layer=pad_layer,
        centered=True,
    )
    right_pad.move((bridge_length / 2 + pad_length / 2, 0))

    # Port configuration data
    port_configs = [
        {
            "name": "o1",
            "center": (0, bridge_width / 2),
            "width": bridge_width,
            "orientation": 90,
            "layer": bridge_layer,
        },
        {
            "name": "o2",
            "center": (0, -bridge_width / 2),
            "width": bridge_width,
            "orientation": 270,
            "layer": bridge_layer,
        },
        {
            "name": "e1",
            "center": (-bridge_length / 2 - pad_length, 0),
            "width": pad_width,
            "orientation": 180,
            "layer": pad_layer,
            "port_type": "electrical",
        },
        {
            "name": "e2",
            "center": (bridge_length / 2 + pad_length, 0),
            "width": pad_width,
            "orientation": 0,
            "layer": pad_layer,
            "port_type": "electrical",
        },
    ]

    # Add all ports using the configuration data
    for config in port_configs:
        c.add_port(**config)

    c.rotate(90)

    return c


def cpw_with_airbridges(
    airbridge_spacing: float = 100.0,
    airbridge_padding: float = 10.0,
    bridge_component: Component | None = None,
    **cpw_kwargs,
) -> gf.CrossSection:
    """Create a coplanar waveguide cross-section with airbridges.

    This function creates a CPW cross-section that includes airbridges placed
    at regular intervals along the transmission line. This is useful for
    preventing slot mode propagation and reducing crosstalk in quantum circuits.

    Args:
        airbridge_spacing: Distance between airbridge centers in µm.
        airbridge_padding: Minimum distance from path start to first airbridge in µm.
        bridge_component: Custom airbridge component. If None, uses default airbridge.
        **cpw_kwargs: Additional arguments passed to the coplanar_waveguide function.

    Returns:
        CrossSection with airbridges for use in routing functions.
    """
    if bridge_component is None:
        bridge_component = airbridge()

    # Create base CPW cross-section
    base_xs = tech.coplanar_waveguide(**cpw_kwargs)

    # Create ComponentAlongPath for airbridges
    component_along_path = gf.ComponentAlongPath(
        component=bridge_component,
        spacing=airbridge_spacing,
        padding=airbridge_padding,
    )

    # Create a copy with airbridges using Pydantic model_copy
    return base_xs.model_copy(update={"components_along_path": (component_along_path,)})


if __name__ == "__main__":
    # Example usage and testing
    from qpdk import PDK

    PDK.activate()

    # Create and display a single airbridge
    bridge = airbridge()
    bridge.show()
