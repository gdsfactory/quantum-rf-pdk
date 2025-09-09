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
# # Write GDS with Hello World
#
# This sample demonstrates creating a simple GDS layout with text and geometric shapes.

# %%
from __future__ import annotations

import gdsfactory as gf

from qpdk import LAYER

# %% [markdown]
# ## Sample Function
#
# Creates a component with 'Hello world' text and a rectangle positioned relative to each other.


# %%
@gf.cell
def sample0_hello_world() -> gf.Component:
    """Returns a component with 'Hello world' text and a rectangle."""
    c = gf.Component()
    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=LAYER.M1_DRAW))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=LAYER.M1_DRAW))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=LAYER.M1_DRAW))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    ref3.rotate(90)
    return c
