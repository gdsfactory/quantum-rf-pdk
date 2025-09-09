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
# # Write GDS with Remove Layers
#
# This sample demonstrates how to remove specific layers from a component after creation.

# %%
import gdsfactory as gf

from qpdk import LAYER

# %% [markdown]
# ## Sample Function
#
# Creates a component with text and rectangles, then removes the etch layer to show layer manipulation.


# %%
@gf.cell
def sample2_remove_layers() -> gf.Component:
    """Returns a component with 'Hello world' text and a rectangle."""
    c = gf.Component()

    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=LAYER.M1_ETCH))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=LAYER.M1_DRAW))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=LAYER.M1_DRAW))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    c.flatten()
    return c.remove_layers(layers=["M1_ETCH"])
