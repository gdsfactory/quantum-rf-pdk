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

from qpdk import PDK, cells, tech
from qpdk.cells.helpers import apply_additive_metals, fill_magnetic_vortices

# %% [markdown]
# ## 3D Sample Function
#
# Creates a transmon coupled to a resonator with a chip edge for 3D visualization.


# %%
@gf.cell
def sample_to_3d() -> gf.Component:
    """Returns a transmon with a resonator and chip edge for 3D visualization.

    This function demonstrates the full workflow of adding logical components,
    filling with magnetic vortices, adding a simulation area, and finally
    applying the additive metal transformation for mask generation and 3D visualization.
    """
    c = gf.Component()

    # Create the transmon with resonator
    tr = c << cells.double_pad_transmon_with_resonator()

    # Add a chip edge around the component
    bbox = tr.bbox()
    margin = 100.0
    chip_size = (bbox.width() + 2 * margin, bbox.height() + 2 * margin)

    ce = c << cells.chip_edge(size=chip_size, width=20.0)
    ce.center = tr.center

    # fill_magnetic_vortices adds small etch holes for magnetic flux trapping.
    # It returns a new component cell.
    c = fill_magnetic_vortices(c)

    # We must use .dup() because apply_additive_metals flattens the component and removes layers,
    # and we want to perform these modifications on a fresh copy to avoid LockedErrors
    # since we are inside a @cell-decorated function and the result of fill_magnetic_vortices is also a cell.
    c = c.dup()

    # For 3D visualization, we need to add the SIM_AREA layer which acts as the bulk metal/substrate
    # from which etched regions are subtracted in the LayerStack.
    # We match the SIM_AREA to the chip edge area.
    bbox_total = c.bbox()
    c.add_polygon(
        [
            (bbox_total.left, bbox_total.bottom),
            (bbox_total.right, bbox_total.bottom),
            (bbox_total.right, bbox_total.top),
            (bbox_total.left, bbox_total.top),
        ],
        layer=tech.LAYER.SIM_AREA,
    )

    # apply_additive_metals generates the final negative mask on M1_ETCH by subtracting M1_DRAW.
    return apply_additive_metals(c)


# %% [markdown]
# ## Visualization
#
# Shows the component in 2D (KLayout) and 3D.

# %%
if __name__ == "__main__":
    PDK.activate()

    c = sample_to_3d()
    c.show()
    s = c.to_3d(layer_stack=tech.LAYER_STACK_NO_VACUUM)
    s.show()
