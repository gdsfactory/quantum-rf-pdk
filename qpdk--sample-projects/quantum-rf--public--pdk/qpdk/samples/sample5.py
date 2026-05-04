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
# # Path Creation Sample
#
# This sample demonstrates creating complex curved paths using gdsfactory's path functionality.

# %%
import gdsfactory as gf

# %% [markdown]
# ## Sample Function
#
# Creates a component with a path made of different segments including arcs, straight lines, and euler bends.


# %%
@gf.cell
def sample5_path():
    """Returns a component with a path made of different segments."""
    p = gf.Path()
    p += gf.path.arc(radius=10, angle=90)  # Circular arc
    p += gf.path.straight(length=10)  # Straight section
    p += gf.path.euler(radius=3, angle=-90)  # Euler bend (aka "racetrack" curve)
    p += gf.path.straight(length=40)
    p += gf.path.arc(radius=8, angle=-45)
    p += gf.path.straight(length=10)
    p += gf.path.arc(radius=8, angle=45)
    p += gf.path.straight(length=10)
    return p.extrude(layer=(3, 0), width=1.5)
