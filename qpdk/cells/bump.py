"""Indium bump components for 3D integration."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component

from qpdk.tech import LAYER


@gf.cell
def indium_bump(diameter: float = 15.0) -> Component:
    """Creates an indium bump component for 3D integration.

    See :cite:`rosenberg3DIntegratedSuperconducting2017` for details.

    Args:
        diameter: Diameter of the indium bump in µm.

    Returns:
        A gdsfactory Component representing the indium bump.
    """
    c = Component()
    circle = gf.components.circle(radius=diameter / 2, layer=LAYER.IND)
    ref = c.add_ref(circle)
    ref.move((0, 0))
    c.add_port(
        name="center",
        center=(
            0,
            0,
        ),
        orientation=0,
        layer=LAYER.IND,
        width=diameter,
    )
    return c


if __name__ == "__main__":
    bump = indium_bump()
    bump.show()
