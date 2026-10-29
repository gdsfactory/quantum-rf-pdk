"""Additive metal tests."""

import pytest
from gdsfactory import Component
from gdsfactory.typings import Layer
from klayout.db import Region

from qpdk import PDK
from qpdk.cells.transmon import flipmon_with_bbox
from qpdk.cells.tsv import tsv_transition_double_sided
from qpdk.helper import layerenum_to_tuple
from qpdk.tech import CHIP_LAYERS, LAYER, Chip
from qpdk.utils import apply_additive_metals, invert_mask_polarity

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


@pytest.mark.parametrize("chip", ["bottom", "top"])
def test_apply_additive_metals_backside(chip: Chip) -> None:
    """Backside draw shapes fold into the backside etch layer of each chip."""
    PDK.activate()
    layers = CHIP_LAYERS[chip]
    draw, etch = (
        layerenum_to_tuple(layers.back_draw),
        layerenum_to_tuple(layers.back_etch),
    )

    c = Component()
    c << tsv_transition_double_sided(chip=chip)
    assert {draw, etch} <= set(c.layers)

    polygons = apply_additive_metals(c).get_polygons(by="tuple")
    assert not polygons.get(draw)
    assert polygons.get(etch)


@pytest.mark.parametrize("chip", ["bottom", "top"])
def test_invert_mask_polarity_backside(chip: Chip) -> None:
    """Inversion swaps the backside draw and etch layers of each chip."""
    PDK.activate()
    layers = CHIP_LAYERS[chip]
    c = Component()
    c << tsv_transition_double_sided(chip=chip)
    dbu = c.kcl.dbu
    bbox = Region(c.bbox().to_itype(dbu))

    def region(component: Component, layer: Layer) -> Region:
        index = component.kcl.layer(*layer)
        return Region(component.begin_shapes_rec(index)).merged()

    inverted = invert_mask_polarity(c)
    for draw, etch in (
        (layers.front_draw, layers.front_etch),
        (layers.back_draw, layers.back_etch),
    ):
        assert (region(inverted, draw) ^ (bbox - region(c, etch))).is_empty()
        assert (region(inverted, etch) ^ (bbox - region(c, draw))).is_empty()
    assert (region(inverted, layers.tsv) ^ region(c, layers.tsv)).is_empty()
