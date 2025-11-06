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
# # Resonator Test Chip
#
# This example demonstrates creating a resonator test chip for characterizing superconducting microwave resonators.
#
# The design is inspired by {cite:p}`norrisImprovedParameterTargeting2024`.

# %%
import gdsfactory as gf
import numpy as np

from qpdk import tech
from qpdk.cells.chip import chip_edge
from qpdk.cells.helpers import fill_magnetic_vortices
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import resonator_coupled
from qpdk.cells.waveguides import straight
from qpdk.tech import (
    coplanar_waveguide,
    route_single_cpw,
    route_single_sbend,
)

# %% [markdown]
# ## Resonator Test Chip Function
#
# Creates a test chip with two probelines and multiple resonators for characterization.


# %%
@gf.cell
def resonator_test_chip_python(
    probeline_length: float = 9000.0,
    probeline_separation: float = 1000.0,
    resonator_length: float = 4000.0,
    coupling_length: float = 200.0,
    coupling_gap: float = 16.0,
) -> gf.Component:
    """Creates a resonator test chip with two probelines and 16 resonators.

    The chip features two horizontal probelines running west to east, each with
    launchers on both ends. Eight quarter-wave resonators are coupled to each
    probeline, with systematically varied cross-section parameters for
    characterization studies.

    Args:
        probeline_length: Length of each probeline in µm.
        probeline_separation: Vertical separation between probelines in µm.
        resonator_length: Length of each resonator in µm.
        coupling_length: Length of coupling region between resonator and probeline in µm.
        coupling_gap: Gap between resonator and probeline for coupling in µm.

    Returns:
        Component: A gdsfactory component containing the complete test chip layout.
    """
    c = gf.Component()

    # Create different cross-sections for resonators with systematic parameter variation
    # 8 different combinations of width and gap for each probeline
    width_values = np.linspace(8, 30, 8, dtype=int)
    gap_values = np.linspace(6, 20, 8, dtype=int)

    n_resonators = len(width_values)
    resonator_cross_sections = [
        coplanar_waveguide(width=width_values[i], gap=gap_values[i])
        for i in range(n_resonators)
    ]

    # Standard cross-section for probelines
    probeline_xs = coplanar_waveguide(width=10, gap=6)

    probeline_y_positions = [0, probeline_separation]

    for probeline_idx, y_pos in enumerate(probeline_y_positions):
        # Add launchers at both ends
        launcher_west = c.add_ref(launcher())
        launcher_west.move((0, y_pos))
        launcher_east = c.add_ref(launcher())  # Create some probeline straight
        launcher_east.mirror_x()
        launcher_east.move((probeline_length, y_pos))

        # Add resonators along the probeline
        resonator_spacing = probeline_length / 9  # Space for 8 resonators

        previous_port = launcher_west.ports["o1"]
        for res_idx in range(n_resonators):
            # Calculate resonator position along probeline
            x_position = (res_idx + 1) * resonator_spacing

            # Create resonator with unique cross-section
            coupled_resonator = resonator_coupled(
                length=resonator_length,
                meanders=6,
                cross_section=resonator_cross_sections[res_idx],
                open_start=True,
                open_end=False,  # Quarter-wave resonator
                cross_section_non_resonator=probeline_xs,
                coupling_straight_length=coupling_length,
                coupling_gap=coupling_gap,
            )
            resonator_ref = c.add_ref(coupled_resonator)
            # Position resonator above probeline
            if probeline_idx != 0:
                resonator_ref.mirror_y()

            resonator_ref.move((x_position - resonator_ref.size_info.width / 2, y_pos))
            gf.logger.debug(f"Added resonator {res_idx} at x={x_position} µm")

            if res_idx == 0:
                # Add some straight before connecting the first resonator
                first_straight_ref = c.add_ref(
                    straight(length=200.0, cross_section=probeline_xs)
                )
                first_straight_ref.connect("o1", resonator_ref.ports["coupling_o1"])
                route_single_sbend(
                    c,
                    port1=previous_port,
                    port2=first_straight_ref.ports["o2"],
                    cross_section=probeline_xs,
                )
            else:
                route_single_cpw(
                    c,
                    port1=previous_port,
                    port2=resonator_ref.ports["coupling_o1"],
                    cross_section=probeline_xs,
                )

            previous_port = resonator_ref.ports["coupling_o2"]

        # Add some straight before connecting to the final launcher
        final_straight_ref = c.add_ref(
            straight(length=400.0, cross_section=probeline_xs)
        )
        final_straight_ref.connect("o1", previous_port)

        # Connect final launcher to probeline
        route_single_sbend(
            c,
            port1=final_straight_ref.ports["o2"],
            port2=launcher_east.ports["o1"],
            cross_section=probeline_xs,
        )

    return c


# %% [markdown]
# ## Filled Resonator Test Chip
#
# Version of the test chip with magnetic vortex trapping holes in the ground plane.


# %%
@gf.cell
def filled_resonator_test_chip() -> gf.Component:
    """Creates a resonator test chip filled with magnetic vortex trapping holes.

    This version includes the complete resonator test chip layout with additional
    ground plane holes to trap magnetic vortices, improving the performance of
    superconducting quantum circuits. Includes chip edge components with extra
    y-padding to keep resonators away from the chip edges.

    Returns:
        Component: Test chip with ground plane fill patterns and chip edges.
    """
    c = gf.Component()
    test_chip = resonator_test_chip_python()
    c << test_chip
    chip_edge_ref = c << chip_edge(
        size=(test_chip.xsize + 200, test_chip.ysize + 800),
        width=100.0,
        layer=tech.LAYER.M1_ETCH,
    )
    chip_edge_ref.move((test_chip.xmin - 100, test_chip.ymin - 400))
    return fill_magnetic_vortices(
        component=c,
        rectangle_size=(15.0, 15.0),
        gap=70.0,
        stagger=2,
    )


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # Create and display the filled version
    filled_chip = filled_resonator_test_chip()
    filled_chip.show()
