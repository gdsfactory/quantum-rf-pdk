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
# # Braided Giant Atoms Chip
#
# Two giant transmons share one waveguide in the braided configuration of
# {cite:p}`kannanWaveguideQuantumElectrodynamics2020`: the coupling points are
# ordered $x_{\text{a}1} < x_{\text{b}1} < x_{\text{a}2} < x_{\text{b}2}$ with
# equal waveguide length $\Delta x$ between neighbours. With
# $\Delta x = \lambda/4$ each qubit decouples from the waveguide at the design
# frequency while the two qubits remain coupled through it.
#
# Launchers on both ends give access to the transmission $S_{21}$. The
# frequency-dependent decay rate of one giant atom is modelled by
# {func}`qpdk.models.giant_atom.giant_atom_transmon` and
# {func}`qpdk.models.giant_atom.giant_atom_decay_rate_and_lamb_shift`.

# %%
import gdsfactory as gf

from qpdk import PDK, cells, tech
from qpdk.cells.chip import chip_edge
from qpdk.cells.giant_atom import braided_giant_atoms


# %%
@gf.cell
def giant_atom_chip(
    segment_length: float = 6550.0,
    launcher_distance: float = 400.0,
    edge_margin: float = 300.0,
) -> gf.Component:
    """Returns two braided giant atoms between two launchers on a chip.

    Args:
        segment_length: Waveguide length between neighbouring coupling points
            in µm.
        launcher_distance: Waveguide length between each launcher and the
            giant atoms in µm.
        edge_margin: Clearance between the circuit and the chip edge in µm.
    """
    c = gf.Component()
    atoms = c << braided_giant_atoms(segment_length=segment_length)

    launcher_in = c << cells.launcher()
    launcher_in.connect("o1", atoms.ports["o1"])
    launcher_in.dmovex(-launcher_distance)
    launcher_out = c << cells.launcher()
    launcher_out.connect("o1", atoms.ports["o2"])
    launcher_out.dmovex(launcher_distance)

    tech.route_bundle_cpw(c, [launcher_in.ports["o1"]], [atoms.ports["o1"]])
    tech.route_bundle_cpw(c, [atoms.ports["o2"]], [launcher_out.ports["o1"]])

    bbox = c.dbbox()
    edge = c << chip_edge(
        size=(bbox.width() + 2 * edge_margin, bbox.height() + 2 * edge_margin)
    )
    edge.dmove((bbox.left - edge_margin, bbox.bottom - edge_margin))
    return c


# %%
if __name__ == "__main__":
    PDK.activate()
    giant_atom_chip().show()
