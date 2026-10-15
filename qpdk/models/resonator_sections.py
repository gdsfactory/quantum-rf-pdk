"""SAX models for the physical sections of a coupled resonator."""

import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.resonator import resonator_coupled
from qpdk.models.waveguides import straight


def resonator_coupling_region(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 300.0,
    coupling_straight_length: float = 200.0,
    coupling_gap: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
    cross_section_non_resonator: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """Model the bus and the open-ended resonator segment coupled to it.

    The section has two bus ports and two resonator ports. Its lumped mutual
    capacitance and transmission-line sections match the existing whole-device
    model; the meander and far-end short are connected outside this section.
    """
    return resonator_coupled(
        f=f,
        length=length,
        coupling_gap=coupling_gap,
        coupling_straight_length=coupling_straight_length,
        cross_section=cross_section,
        cross_section_non_resonator=cross_section_non_resonator,
        open_start=True,
        open_end=False,
    )


def resonator_meander(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 3700.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """Model the meander by its total CPW electrical length."""
    return straight(f=f, length=length, cross_section=cross_section)
