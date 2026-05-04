# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Routing with Airbridges Example
#
# This example demonstrates how to use airbridges in quantum circuit routing to prevent slot mode propagation and reduce crosstalk between transmission lines.

# %%
import gdsfactory as gf

from qpdk import PDK, cells, tech

# %% [markdown]
# ## Main Example
#
# Creates two launcher components and routes between them using coplanar waveguides with airbridges.

# %%
if __name__ == "__main__":
    PDK.activate()

    # Create a component to demonstrate routing with airbridges
    c = gf.Component()

    # Create two launcher components for connection
    launcher1 = c << cells.launcher()
    launcher1.move((-300, 0))
    launcher2 = c << cells.launcher()
    launcher2.rotate(270)

    # Position the second launcher in a simpler way
    launcher2.move((1000, 1000))

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
        waypoints=[(500, 0), (500, 800)],
    )

    c.show()

    print("Routing with airbridges example created successfully!")
