"""Pack."""

import gdsfactory as gf


@gf.cell
def sample3_grid():
    """Returns a component with a grid of text elements."""
    t1 = gf.components.text("1", layer="M1_DRAW")
    t2 = gf.components.text("2", layer="M1_DRAW")
    t3 = gf.components.text("3", layer="M1_DRAW")
    t4 = gf.components.text("4", layer="M1_DRAW")
    t5 = gf.components.text("5", layer="M1_DRAW")
    t6 = gf.components.text("6", layer="M1_DRAW")

    return gf.grid([t1, t2, t3, t4, t5, t6], shape=(2, 3), spacing=(10, 10))
