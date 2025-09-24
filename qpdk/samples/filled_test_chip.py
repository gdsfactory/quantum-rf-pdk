# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% tags=["hide-input", "hide-output"]
from functools import partial
from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

from qpdk import PDK, tech
from qpdk.cells.chip import chip_edge
from qpdk.cells.helpers import apply_additive_metals, fill_magnetic_vortices
from qpdk.helper import layerenum_to_tuple

# %% [markdown]
# # Filled Qubit Test Chip Example
#
# This example demonstrates creating a qubit test chip filled with magnetic vortex trapping rectangles.
#
# The design roughly corresponds to the sample described in Tuokkola et al. "Methods to achieve near-millisecond coherence times in superconducting quantum circuits" (2025).


# %% [markdown]
# ## Filled Qubit Test Chip Function
#
# Creates a qubit test chip from a YAML configuration and fills it with magnetic vortex trapping rectangles.
#
# See the YAML file for the original test chip layout.

# %% [rst]
#
# .. note::
#
#    See the YAML file for the original test chip layout.
#
# .. include:: ./qubit_test_chip.pic.yml
#


# %%
@gf.cell
def filled_qubit_test_chip(
    yaml_path: str | Path = Path(__file__).parent / "qubit_test_chip.pic.yml",
):
    """Returns a qubit test chip filled with magnetic vortex trapping rectangles.

    Rouhly corresponds to the sample in :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    c = gf.Component()
    test_chip = from_yaml(
        yaml_path,
        routing_strategies=tech.routing_strategies,
    )
    c << fill_magnetic_vortices(
        component=test_chip,
        rectangle_size=(15.0, 15.0),
        gap=70.0,
        stagger=2,
    )
    # Add chip edge component
    chip_edge_ref = c << chip_edge(
        size=(test_chip.xsize + 230, test_chip.ysize + 200),
        width=100.0,
        layer=tech.LAYER.M1_ETCH,
    )
    # Position chip edge to align with test chip bounds
    chip_edge_ref.move((test_chip.xmin - 100, test_chip.ymin - 100))
    # Flip-chip
    if any(
        layerenum_to_tuple(layer_enum) in c.layers
        for layer_enum in (tech.LAYER.M2_DRAW, tech.LAYER.M2_ETCH)
    ):
        chip_edge_ref = c << chip_edge(
            size=(test_chip.xsize + 230, test_chip.ysize + 200),
            width=100.0,
            layer=tech.LAYER.M2_ETCH,
        )
        # Position chip edge to align with test chip bounds
        chip_edge_ref.move((test_chip.xmin - 100, test_chip.ymin - 100))
        c << fill_magnetic_vortices(
            component=test_chip,
            rectangle_size=(15.0, 15.0),
            gap=70.0,
            stagger=2,
            exclude_layers=[
                (tech.LAYER.M2_ETCH, 80),
                (tech.LAYER.M2_DRAW, 80),
            ],
            fill_layer=tech.LAYER.M2_ETCH,
        )

    # Get final 'negative' layout
    return apply_additive_metals(c)


# %% [markdown]
# ## Filled Flipmon Test Chip Function
#
# Creates a flipmon test chip from a similar YAML configuration and fills it as well.
# This version uses flipmon qubits for flip-chip applications.
# See {cite:p}`liVacuumgapTransmonQubits2021` for more details.


# %%
filled_flipmon_test_chip = partial(
    filled_qubit_test_chip, Path(__file__).parent / "flipmon_test_chip.pic.yml"
)

# %% [markdown]
# ## Examples

# %%
if __name__ == "__main__":
    PDK.activate()

    # Show original filled qubit test chip
    filled_qubit_test_chip().show()

    # %%

    # Show new filled flipmon test chip
    filled_flipmon_test_chip().show()
