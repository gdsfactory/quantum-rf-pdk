"""Sample component with filled quarter wave resonator for magnetic vortex trapping."""

import gdsfactory as gf

from qpdk.cells.helpers import fill_magnetic_vortices
from qpdk.cells.resonator import resonator_quarter_wave


@gf.cell
def sample_filled_quarter_wave_resonator():
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
    filled_resonator = fill_magnetic_vortices(
        component=resonator,
        rectangle_size=(15.0, 15.0),
        gap=15.0,
        keepout_margin=80.0,
    )
    
    return filled_resonator


if __name__ == "__main__":
    # Example usage and testing
    from qpdk import PDK

    PDK.activate()

    # Create and display the filled resonator
    c = sample_filled_quarter_wave_resonator()
    c.show()