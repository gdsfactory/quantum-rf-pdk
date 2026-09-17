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
# # Write GDS with Sample Connections
#
# This sample demonstrates how to connect waveguides sequentially to create a longer path.

# %%
import gdsfactory as gf

from qpdk import cells, tech

# %% [markdown]
# ## Sample Function
#
# Creates a component with three connected waveguides of increasing length.


# %%
@gf.cell
def sample1_connect() -> gf.Component:
    """Returns a component with connected waveguides."""
    c = gf.Component()
    cross_section = tech.cpw(width=1)
    wg1 = c << cells.straight(length=1, cross_section=cross_section)
    wg2 = c << cells.straight(length=2, cross_section=cross_section)
    wg3 = c << cells.straight(length=3, cross_section=cross_section)

    wg2.connect(port="o1", other=wg1["o2"])
    wg3.connect(port="o1", other=wg2["o2"])
    return c
