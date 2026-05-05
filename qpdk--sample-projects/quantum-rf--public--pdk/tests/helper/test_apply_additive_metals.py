"""Additive metal tests."""

from gdsfactory import Component

from qpdk.cells.helpers import apply_additive_metals
from qpdk.cells.transmon import flipmon_with_bbox
from qpdk.helper import layerenum_to_tuple
from qpdk.tech import LAYER

ADDITIVE_LAYERS = {
    layerenum_to_tuple(layer_enum) for layer_enum in (LAYER.M1_DRAW, LAYER.M2_DRAW)
}


def test_apply_additive_metals_m1_m2():
    from qpdk import PDK

    PDK.activate()

    c = Component()
    c << flipmon_with_bbox()

    assert ADDITIVE_LAYERS.issubset(c.layers)
    c = apply_additive_metals(c)
    assert not ADDITIVE_LAYERS.issubset(c.layers)


if __name__ == "__main__":
    test_apply_additive_metals_m1_m2()
