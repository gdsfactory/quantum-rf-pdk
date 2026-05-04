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
# # Filled Quarter Wave Resonator Sample
#
# This sample demonstrates how to use the fill_magnetic_vortices helper function to add small rectangles that trap magnetic vortices in superconducting quantum circuits.

# %%
import gdsfactory as gf

from qpdk.cells.helpers import fill_magnetic_vortices
from qpdk.cells.resonator import resonator_quarter_wave
from qpdk.tech import LAYER

# %% [markdown]
# ## Filled Quarter Wave Resonator Function
#
# Creates a quarter wave resonator and fills it with magnetic vortex trapping rectangles.


# %%
@gf.cell
def filled_quarter_wave_resonator():
    """Returns a quarter wave resonator filled with magnetic vortex trapping rectangles.

    This sample demonstrates how to use the fill_magnetic_vortices helper function
    to add small rectangles that trap magnetic vortices in superconducting quantum
    circuits.

    Returns:
        Component: A quarter wave resonator with fill rectangles for vortex trapping.
    """
    # Create a quarter wave resonator
    resonator = resonator_quarter_wave(length=2000.0)

    # Fill it with magnetic vortex trapping rectangles
    return fill_magnetic_vortices(
        component=resonator,
        rectangle_size=(15.0, 15.0),
        gap=15.0,
        exclude_layers=[
            (LAYER.M1_ETCH, 20),
        ],
    )


# %% [markdown]
# ## Example Usage
#
# Demonstrates how to create and display the filled resonator.

# %%
if __name__ == "__main__":
    # Example usage and testing
    from qpdk import PDK

    PDK.activate()

    # Create and display the filled resonator
    c = filled_quarter_wave_resonator()
    c.show()
