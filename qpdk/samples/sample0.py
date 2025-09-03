"""Write GDS with hello world."""

from __future__ import annotations

import gdsfactory as gf

from qpdk import LAYER


@gf.cell
def sample0_hello_world() -> gf.Component:
    """Returns a component with 'Hello world' text and a rectangle."""
    c = gf.Component()
    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=LAYER.M1_DRAW))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=LAYER.M1_DRAW))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=LAYER.M1_DRAW))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    ref3.rotate(90)
    return c
