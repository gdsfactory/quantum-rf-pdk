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
# The design is inspired by Norris, G.J., Michaud, L., Pahl, D. et al. "Improved parameter targeting in 3D-integrated superconducting circuits through a polymer spacer process." EPJ Quantum Technol. 11, 5 (2024). https://doi.org/10.1140/epjqt/s40507-023-00213-x

# %%
import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from qpdk.cells.helpers import fill_magnetic_vortices
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import resonator_coupled
from qpdk.cells.waveguides import straight
from qpdk.tech import coplanar_waveguide

# %% [markdown]
# ## Resonator Test Chip Function
#
# Creates a test chip with two probelines and multiple resonators for characterization.

# %%
@gf.cell
def resonator_test_chip(
    probeline_length: float = 8000.0,
    probeline_separation: float = 2000.0,
    resonator_length: float = 4000.0,
    coupling_length: float = 300.0,
    coupling_gap: float = 12.0,
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
    width_values = [8, 9, 10, 11, 12, 13, 14, 15]  # µm
    gap_values = [4, 5, 6, 7, 8, 9, 10, 11]  # µm
    
    # Create cross-sections for resonators
    resonator_cross_sections = []
    for i in range(8):
        xs = coplanar_waveguide(width=width_values[i], gap=gap_values[i])
        resonator_cross_sections.append(xs)
    
    # Standard cross-section for probelines
    probeline_xs = coplanar_waveguide(width=10, gap=6)
    
    # Create probelines with launchers
    probeline_y_positions = [0, probeline_separation]
    
    for probeline_idx, y_pos in enumerate(probeline_y_positions):
        # Create probeline
        probeline = straight(length=probeline_length, cross_section=probeline_xs)
        probeline_ref = c.add_ref(probeline)
        probeline_ref.move((0, y_pos))
        
        # Add launchers at both ends
        launcher_west = c.add_ref(launcher())
        launcher_east = c.add_ref(launcher())
        
        # Connect launchers to probeline
        launcher_west.connect("o1", probeline_ref.ports["o1"])
        launcher_east.connect("o1", probeline_ref.ports["o2"])
        
        # Add resonators along the probeline
        resonator_spacing = probeline_length / 9  # Space for 8 resonators
        
        for res_idx in range(8):
            # Calculate resonator position along probeline
            x_position = (res_idx + 1) * resonator_spacing
            
            # Create resonator with unique cross-section
            resonator_params = {
                "length": resonator_length,
                "meanders": 6,
                "cross_section": resonator_cross_sections[res_idx],
                "open_start": False,
                "open_end": True,  # Quarter-wave resonator
            }
            
            coupled_resonator = resonator_coupled(
                resonator_params=resonator_params,
                cross_section_non_resonator=probeline_xs,
                coupling_straight_length=coupling_length,
                coupling_gap=coupling_gap,
            )
            
            resonator_ref = c.add_ref(coupled_resonator)
            
            # Position resonator perpendicular to probeline
            # Resonators on top probeline extend upward, bottom probeline extend downward
            if probeline_idx == 0:  # Bottom probeline
                resonator_ref.rotate(90)
                resonator_ref.move((x_position, y_pos))
            else:  # Top probeline
                resonator_ref.rotate(-90)
                resonator_ref.move((x_position, y_pos))
    
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
    superconducting quantum circuits.
    
    Returns:
        Component: Test chip with ground plane fill patterns.
    """
    chip = resonator_test_chip()
    
    return fill_magnetic_vortices(
        component=chip,
        rectangle_size=(15.0, 15.0),
        gap=15.0,
        stagger=5.0,
    )


if __name__ == "__main__":
    from qpdk import PDK
    
    PDK.activate()
    
    # Create and display the test chip
    chip = resonator_test_chip()
    chip.show()
    
    # Create and display the filled version
    filled_chip = filled_resonator_test_chip()
    filled_chip.show()