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
# # Sample 3D Visualization
#
# This sample demonstrates how to create a 3D visualization of a layout with different cross-sections.

# %%
import gdsfactory as gf

from qpdk import PDK, cells

# %% [markdown]
# ## 3D Sample Function
#
# Creates a grid of straights with different cross-sections.


# %%
@gf.cell
def sample_to_3d() -> gf.Component:
    """Returns a component with three different straight waveguides in a grid."""
    c1 = cells.straight(cross_section="strip", length=5)
    c2 = cells.straight(cross_section="microstrip", length=5)
    c3 = cells.straight(cross_section="coplanar_waveguide", length=5)

    return gf.grid([c1, c2, c3])


# %% [markdown]
# ## Visualization
#
# Shows the component in 2D (KLayout) and 3D.

# %%
if __name__ == "__main__":
    PDK.activate()

    c = sample_to_3d()
    c.show()
    s = c.to_3d()
    s.show()
