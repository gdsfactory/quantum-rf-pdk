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
# # Cross-Section Sample
#
# This sample demonstrates creating custom cross-sections with multiple layers and offsets.

# %%
import gdsfactory as gf

# %% [markdown]
# ## Sample Function
#
# Creates a component with a path using a custom cross-section that includes multiple layers with different widths and offsets.


# %%
@gf.cell
def sample6_cross_section():
    """Returns a component with a path made of different segments."""
    p = gf.path.straight()

    # Add a few "sections" to the cross-section
    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("in", "out"))
    s1 = gf.Section(width=2, offset=2, layer=(2, 0))
    s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
    x = gf.CrossSection(sections=(s0, s1, s2))

    return gf.path.extrude(p, cross_section=x)
