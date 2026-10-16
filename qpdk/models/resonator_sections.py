"""SAX models for the physical sections of a coupled resonator."""

import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.waveguides import straight


def resonator_meander(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 3700.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """Model the meander by its total CPW electrical length.

    Bends are approximated as straight CPW of equal length. The layout's
    meander count, bend specification, and end-bend setting do not alter this
    electrical model, matching the pre-section whole-device model.

    Returns:
        Two-port SAX S-parameters for the meander path.
    """
    return straight(f=f, length=length, cross_section=cross_section)
