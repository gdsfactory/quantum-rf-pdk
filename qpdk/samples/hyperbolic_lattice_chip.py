# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///
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
# # Hyperbolic Lattice Chip
#
# A {7,3} hyperbolic lattice of CPW edge resonators in the scheme of
# {cite:p}`kollarHyperbolicLatticesCircuit2019`: one meandered resonator per
# edge of the heptagon tiling, a three-pin capacitive coupler on every vertex,
# shorted stubs on the free boundary pins and two probe launchers.
#
# Near the full-wave mode of the resonators the photons hop on the line graph
# of the tiling with $t < 0$, so the spectrum has a flat band at
# $\omega_0 - 2|t|$. The SAX model
# {func}`qpdk.models.hyperbolic.hyperbolic_lattice` evaluates the same circuit.

# %%
import gdsfactory as gf

from qpdk import PDK
from qpdk.cells.chip import chip_edge
from qpdk.cells.hyperbolic import hyperbolic_lattice


# %%
@gf.cell
def hyperbolic_lattice_chip(
    p: int = 7,
    q: int = 3,
    shells: int = 1,
    resonator_length: float = 7500.0,
    edge_margin: float = 500.0,
) -> gf.Component:
    """Returns a hyperbolic lattice of edge resonators framed by a chip edge.

    Args:
        p: Polygon side count of the tiling.
        q: Polygons per vertex, i.e. coupler pins.
        shells: Polygon rings around the central polygon.
        resonator_length: Length of every edge resonator in µm.
        edge_margin: Clearance between the lattice and the chip edge in µm.
    """
    c = gf.Component()
    lattice = c << hyperbolic_lattice(
        p=p, q=q, shells=shells, resonator_length=resonator_length
    )
    bbox = lattice.dbbox()
    edge = c << chip_edge(
        size=(bbox.width() + 2 * edge_margin, bbox.height() + 2 * edge_margin)
    )
    edge.move((bbox.left - edge_margin, bbox.bottom - edge_margin))
    c.add_ports(lattice.ports)
    return c


# %%
if __name__ == "__main__":
    PDK.activate()
    hyperbolic_lattice_chip().show()
