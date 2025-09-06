"""Example of routing with airbridges using route_bundle.

This example demonstrates how to use airbridges in quantum circuit routing
to prevent slot mode propagation and reduce crosstalk between transmission lines.
"""

import gdsfactory as gf

from qpdk import PDK, cells, tech

if __name__ == "__main__":
    PDK.activate()
    
    # Create a component to demonstrate routing with airbridges
    c = gf.Component()
    
    # Create two launcher components for connection
    launcher1 = c << cells.launcher()
    launcher2 = c << cells.launcher()
    
    # Position the second launcher in a simpler way
    launcher2.move((400, 0))
    
    # Create CPW cross-section with airbridges
    cpw_with_bridges = cells.cpw_with_airbridges(
        airbridge_spacing=60.0,  # Airbridge every 60 µm
        airbridge_padding=20.0,  # 20 µm from start to first airbridge
    )
    
    # Route between the launchers using the CPW with airbridges
    route = tech.route_bundle(
        c, 
        [launcher1.ports["o1"]], 
        [launcher2.ports["o1"]],
        cross_section=cpw_with_bridges,
    )
    
    # Show the component
    # c.show() # Comment out show to avoid KLive issues in CI
    
    print("Routing with airbridges example created successfully!")
    print(f"Route type: {type(route)}")
    
    # Also demonstrate a simple straight with airbridges
    c2 = gf.Component()
    straight_with_bridges = c2 << gf.components.straight(
        length=200,
        cross_section=cpw_with_bridges,
    )
    # c2.show() # Comment out show to avoid KLive issues in CI
    print("Straight with airbridges created successfully!")