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
# # Packing Algorithm Sample
#
# This sample demonstrates the use of gdsfactory's packing algorithm to efficiently arrange shapes.

# %%
import gdsfactory as gf
import numpy as np

# %% [markdown]
# ## Sample Function
#
# Creates a set of random ellipses and packs them efficiently into a rectangular area.


# %%
@gf.cell
def sample4_pack():
    """Returns a component with a packed set of ellipses."""
    rng = np.random.default_rng()
    ellipses = [
        gf.components.ellipse(radii=tuple(rng.random(2) * n + 2), layer="M1_DRAW")
        for n in range(80)
    ]
    bins = gf.pack(
        ellipses,  # Must be a list or tuple of Components
        spacing=4,  # Minimum distance between adjacent shapes
        aspect_ratio=(1, 1),  # Shape of the box
        max_size=(500, 500),  # Limits the size into which the shapes will be packed
        density=1.05,  # Values closer to 1 pack tighter but require more computation
        sort_by_area=True,  # Pre-sorts the shapes by area
    )
    return bins[0]
