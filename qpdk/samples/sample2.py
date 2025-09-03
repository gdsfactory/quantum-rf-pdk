"""Write GDS with remove layers."""

import gdsfactory as gf

from qpdk import LAYER


@gf.cell
def sample2_remove_layers() -> gf.Component:
    """Returns a component with 'Hello world' text and a rectangle."""
    c = gf.Component()

    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=LAYER.M1_ETCH))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=LAYER.M1_DRAW))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=LAYER.M1_DRAW))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    c.flatten()
    return c.remove_layers(layers=["M1_ETCH"])
