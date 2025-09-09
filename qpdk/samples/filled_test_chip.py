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
# # Filled Qubit Test Chip Example
#
# This example demonstrates creating a qubit test chip filled with magnetic vortex trapping rectangles.
#
# The design roughly corresponds to the sample described in Tuokkola et al. "Methods to achieve near-millisecond coherence times in superconducting quantum circuits" (2025).

# %%
from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

from qpdk import PDK, tech
from qpdk.cells.helpers import fill_magnetic_vortices

# %% [markdown]
# ## Filled Qubit Test Chip Function
#
# Creates a qubit test chip from a YAML configuration and fills it with magnetic vortex trapping rectangles.


# %%
@gf.cell
def filled_qubit_test_chip():
    """Returns a qubit test chip filled with magnetic vortex trapping rectangles.

    Rouhly corresponds to the sample in :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    c = gf.Component()
    test_chip = from_yaml(
        Path(__file__).parent / "qubit_test_chip.pic.yml",
        routing_strategies=tech.routing_strategies,
    )
    c << fill_magnetic_vortices(
        component=test_chip,
        rectangle_size=(15.0, 15.0),
        gap=15.0,
        exclude_layers=[(tech.LAYER.M1_ETCH, 80)],
    )
    return c


# %% [markdown]
# ## Example Usage
#
# Demonstrates how to create and display the filled test chip.

# %%
if __name__ == "__main__":
    PDK.activate()

    filled_qubit_test_chip().show()
