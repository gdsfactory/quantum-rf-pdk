"""Launcher component for RF applications.

This module provides a launcher component for RF wirebonding and probe testing.
The launcher consists of a straight section connected to a tapered section,
transitioning from a large cross-section suitable for probing to a smaller
cross-section for circuit integration.
"""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import straight
from gdsfactory.typings import CrossSectionSpec

from qpdk.cells.waveguides import taper_cross_section
from qpdk.tech import LAYER, coplanar_waveguide, launcher_cross_section_big

LAUNCHER_CROSS_SECTION_BIG = launcher_cross_section_big
LAUNCHER_CROSS_SECTION_SMALL = partial(coplanar_waveguide, etch_layer=LAYER.M1_ETCH)


@gf.cell
def launcher(
    straight_length: float = 200.0,
    taper_length: float = 100.0,
    cross_section_big: CrossSectionSpec = LAUNCHER_CROSS_SECTION_BIG,
    cross_section_small: CrossSectionSpec = "cpw",
) -> Component:
    """Generate an RF launcher pad for wirebonding or probe testing.

    Creates a launcher component consisting of a straight section with large
    cross-section connected to a tapered transition down to a smaller cross-section.
    This design facilitates RF signal access through probes or wirebonds while
    maintaining good impedance matching.

    The default dimensions are taken from :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.

    Args:
        straight_length: Length of the straight, wirebond landing area, section in µm.
        taper_length: Length of the taper section in µm.
        cross_section_big: Cross-section specification for the large end
            of the launcher (probe/wirebond interface).
        cross_section_small: Cross-section specification for the small end
            of the launcher (circuit interface).

    Returns:
        Component: A gdsfactory component containing the complete launcher
            geometry with one output port ("o1") at the small end.
    """
    c = Component()

    # Add the straight section (large cross-section for probe access)
    straight_ref = c << straight(
        length=straight_length, cross_section=cross_section_big
    )

    # Add the tapered transition section
    taper_ref = c << taper_cross_section(
        length=taper_length,
        cross_section1=cross_section_big,
        cross_section2=cross_section_small,
        linear=True,
    )

    # Connect the taper to the straight section
    taper_ref.connect("o1", straight_ref.ports["o2"])

    # Add output port at the small end for circuit connection
    c.add_port(port=taper_ref.ports["o2"], name="o1", cross_section=cross_section_small)

    # Add a port at the large end for reference and simulation purposes
    c.add_port(
        port=straight_ref.ports["o1"],
        name="waveport",
        cross_section=cross_section_big,
    )

    return c


if __name__ == "__main__":
    # Example usage and testing
    from qpdk import PDK

    PDK.activate()

    # Create and display a launcher with default parameters
    c = launcher()
    c.show()
