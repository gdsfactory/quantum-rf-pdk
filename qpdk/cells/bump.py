"""Indium bump components for 3D integration."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component

from qpdk.tech import LAYER


def indium_bump(diameter: float = 15.0) -> Component:
    """Creates an indium bump component for 3D integration.

    Args:
        diameter: Diameter of the indium bump in micrometers.

    Returns:
        A gdsfactory Component representing the indium bump.
    """
    c = Component()
    circle = gf.components.circle(radius=diameter / 2, layer=LAYER.IND)
    ref = c.add_ref(circle)
    ref.move((0, 0))
    for name, layer in (("top", LAYER.M2_DRAW), ("bottom", LAYER.M1_DRAW)):
        c.add_port(name=name, center=(0, 0), orientation=0, layer=layer, width=diameter)
    return c


if __name__ == "__main__":
    bump = indium_bump()
    bump.show()
