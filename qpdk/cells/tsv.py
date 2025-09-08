"""Through-silicon via (TSV) component library for gdsfactory."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component

from qpdk.tech import LAYER


@gf.cell
def tsv(diameter: float = 15.0) -> Component:
    """Creates a Through-silicon via (TSV) component for 3D integration.

    See :cite:`yostSolidstateQubitsIntegrated2020`.

    Args:
        diameter: Diameter of the via in µm.

    Returns:
        A gdsfactory Component representing the TSV.
    """
    c = Component()
    circle = gf.components.circle(radius=diameter / 2, layer=LAYER.TSV)
    ref = c.add_ref(circle)
    ref.move((0, 0))
    c.add_port(
        name="center",
        center=(
            0,
            0,
        ),
        orientation=0,
        layer=LAYER.TSV,
        width=diameter,
    )
    return c


if __name__ == "__main__":
    c = tsv()
    c.show()
