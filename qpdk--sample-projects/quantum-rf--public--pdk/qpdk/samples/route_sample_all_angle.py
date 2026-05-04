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
# # All-Angle Routing Example
#
# This sample demonstrates routing between two components using all-angle routing capabilities.

# %%
import gdsfactory as gf

from qpdk import PDK, cells, tech

# %% [markdown]
# ## Main Example
#
# Creates two interdigital capacitors and routes between them using all-angle routing.

# %%
if __name__ == "__main__":
    PDK.activate()
    c = gf.Component()
    m1 = c << cells.interdigital_capacitor()
    m2 = c << cells.interdigital_capacitor()

    m2.move((400, 200))
    m2.rotate(30)
    route = tech.route_bundle_all_angle(c, [m1.ports["o2"]], [m2.ports["o1"]])
    c.show()
