"""Additive metal tests."""

from gdsfactory import Component

from qpdk import PDK
from qpdk.cells.transmon import flipmon_with_bbox
from qpdk.helper import layerenum_to_tuple
from qpdk.tech import LAYER
from qpdk.utils import apply_additive_metals

ADDITIVE_LAYERS = {
    layerenum_to_tuple(layer_enum) for layer_enum in (LAYER.M1_DRAW, LAYER.M2_DRAW)
}
TARGET_LAYERS = {
    layerenum_to_tuple(layer_enum) for layer_enum in (LAYER.M1_ETCH, LAYER.M2_ETCH)
}


def test_apply_additive_metals_m1_m2():
    PDK.activate()

    c = Component()
    c << flipmon_with_bbox()

    assert ADDITIVE_LAYERS.issubset(c.layers)
    additive_polygons_before = sum(
        len(c.get_polygons(by="tuple").get(layer, [])) for layer in ADDITIVE_LAYERS
    )
    assert additive_polygons_before > 0

    c = apply_additive_metals(c)

    assert not ADDITIVE_LAYERS.issubset(c.layers)
    polygons_after = c.get_polygons(by="tuple")
    additive_polygons_after = sum(
        len(polygons_after.get(layer, [])) for layer in ADDITIVE_LAYERS
    )
    assert additive_polygons_after == 0

    assert TARGET_LAYERS.issubset(c.layers)
    assert sum(len(polygons_after.get(layer, [])) for layer in TARGET_LAYERS) > 0
