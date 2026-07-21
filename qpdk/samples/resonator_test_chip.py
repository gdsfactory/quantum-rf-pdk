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
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import quarter_wave_resonator_coupled
from qpdk.cells.waveguides import straight
from qpdk.logger import logger
from qpdk.tech import (
    coplanar_waveguide,
    route_bundle_cpw,
    route_bundle_sbend,
)
from qpdk.utils import fill_magnetic_vortices

# %% [markdown]
# ## Resonator Test Chip Function
#
# Creates a test chip with two probelines and multiple resonators for characterization.


# %%
@gf.cell
def resonator_test_chip_python(
    probeline_length: float = 9000.0,
    probeline_separation: float = 1000.0,
    resonator_length_range: tuple[float, float] = (3600.0, 5100.0),
    meanders: int = 6,
    coupling_length: float = 200.0,
    coupling_gap: float = 16.0,
) -> gf.Component:
    r"""Creates a resonator test chip with two probelines and 16 resonators.

    The chip features two horizontal probelines running west to east, each with
    launchers on both ends and eight coupled quarter-wave resonators. Resonance
    frequencies are set by the resonator *length* (:math:`f \propto 1/L`), since
    the CPW effective permittivity is nearly geometry independent. The 16 lengths
    are interleaved between the probelines so all resonances are distinct. Inspired
    by :cite:p:`norrisImprovedParameterTargeting2024`.

    Args:
        probeline_length: Length of each probeline in µm.
        probeline_separation: Vertical separation between probelines in µm.
        resonator_length_range: ``(min, max)`` resonator lengths in µm; the 16
            resonator lengths are linearly spaced across this range.
        meanders: Number of meander sections per resonator.
        coupling_length: Length of coupling region between resonator and probeline in µm.
        coupling_gap: Gap between resonator and probeline for coupling in µm.

    Returns:
        Component: A gdsfactory component containing the complete test chip layout.
    """
    c = gf.Component()

    # Shared registered cross-section for every resonator and both probelines.
    cross_section = coplanar_waveguide()

    # 16 linearly spaced lengths, interleaved between the two probelines.
    n_resonators_total = 16
    n_per_probeline = n_resonators_total // 2
    resonator_lengths = np.linspace(*resonator_length_range, n_resonators_total)

    probeline_y_positions = [0, probeline_separation]

    for probeline_idx, y_pos in enumerate(probeline_y_positions):
        # Add launchers at both ends
        launcher_west = c.add_ref(launcher())
        launcher_west.move((0, y_pos))
        launcher_east = c.add_ref(launcher())  # Create some probeline straight
        launcher_east.mirror_x()
        launcher_east.move((probeline_length, y_pos))

        lengths = resonator_lengths[probeline_idx::2]

        # Add resonators along the probeline
        resonator_spacing = probeline_length / (n_per_probeline + 1)

        previous_port = launcher_west.ports["o1"]
        for res_idx in range(n_per_probeline):
            # Calculate resonator position along probeline
            x_position = (res_idx + 1) * resonator_spacing

            # Create quarter-wave resonator with a unique length
            coupled_resonator = quarter_wave_resonator_coupled(
                length=float(lengths[res_idx]),
                meanders=meanders,
                cross_section=cross_section,
                cross_section_non_resonator=cross_section,
                coupling_straight_length=coupling_length,
                coupling_gap=coupling_gap,
            )
            resonator_ref = c.add_ref(coupled_resonator)
            # Position resonator above probeline
            if probeline_idx != 0:
                resonator_ref.mirror_y()

            resonator_ref.move((x_position - resonator_ref.size_info.width / 2, y_pos))
            logger.debug(f"Added resonator {res_idx} at x={x_position} µm")

            if res_idx == 0:
                # Add some straight before connecting the first resonator
                first_straight_ref = c.add_ref(
                    straight(length=200.0, cross_section=cross_section)
                )
                first_straight_ref.connect("o1", resonator_ref.ports["coupling_o1"])
                route_bundle_sbend(
                    c,
                    ports1=[previous_port],
                    ports2=[first_straight_ref.ports["o2"]],
                    cross_section=cross_section,
                )
            else:
                route_bundle_cpw(
                    c,
                    ports1=[previous_port],
                    ports2=[resonator_ref.ports["coupling_o1"]],
                    cross_section=cross_section,
                )

            previous_port = resonator_ref.ports["coupling_o2"]

        # Add some straight before connecting to the final launcher
        final_straight_ref = c.add_ref(
            straight(length=400.0, cross_section=cross_section)
        )
        final_straight_ref.connect("o1", previous_port)

        # Connect final launcher to probeline
        route_bundle_sbend(
            c,
            ports1=[final_straight_ref.ports["o2"]],
            ports2=[launcher_east.ports["o1"]],
            cross_section=cross_section,
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
