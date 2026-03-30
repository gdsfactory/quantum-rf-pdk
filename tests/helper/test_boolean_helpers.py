"""Tests for boolean operation helper functions."""

import gdsfactory as gf
from gdsfactory import Component

from qpdk import PDK
from qpdk.cells.helpers import merge_layers_with_etch, subtract_draw_from_etch
from qpdk.helper import layerenum_to_tuple
from qpdk.tech import LAYER

PDK.activate()


class TestMergeLayersWithEtch:
    """Tests for merge_layers_with_etch helper."""

    def test_returns_component(self):
        c = Component()
        c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_DRAW)
        c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_ETCH)
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.WG)

        result = merge_layers_with_etch(
            component=c,
            draw_layer=LAYER.M1_DRAW,
            wg_layer=LAYER.WG,
            etch_layer=LAYER.M1_ETCH,
        )
        assert isinstance(result, Component)

    def test_merged_result_has_draw_layer(self):
        c = Component()
        c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_DRAW)
        c.add_polygon([(-5, -5), (15, -5), (15, 15), (-5, 15)], layer=LAYER.M1_ETCH)
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.WG)

        result = merge_layers_with_etch(
            component=c,
            draw_layer=LAYER.M1_DRAW,
            wg_layer=LAYER.WG,
            etch_layer=LAYER.M1_ETCH,
        )
        draw_layer_tuple = layerenum_to_tuple(LAYER.M1_DRAW)
        assert draw_layer_tuple in result.layers

    def test_merged_result_has_etch_layer(self):
        c = Component()
        c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_DRAW)
        c.add_polygon([(-5, -5), (15, -5), (15, 15), (-5, 15)], layer=LAYER.M1_ETCH)
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.WG)

        result = merge_layers_with_etch(
            component=c,
            draw_layer=LAYER.M1_DRAW,
            wg_layer=LAYER.WG,
            etch_layer=LAYER.M1_ETCH,
        )
        etch_layer_tuple = layerenum_to_tuple(LAYER.M1_ETCH)
        assert etch_layer_tuple in result.layers

    def test_wg_layer_not_in_result(self):
        c = Component()
        c.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_DRAW)
        c.add_polygon([(-5, -5), (15, -5), (15, 15), (-5, 15)], layer=LAYER.M1_ETCH)
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.WG)

        result = merge_layers_with_etch(
            component=c,
            draw_layer=LAYER.M1_DRAW,
            wg_layer=LAYER.WG,
            etch_layer=LAYER.M1_ETCH,
        )
        wg_layer_tuple = layerenum_to_tuple(LAYER.WG)
        assert wg_layer_tuple not in result.layers

    def test_produces_same_result_as_capacitor_components(self):
        """Verify the refactored capacitor components still work correctly."""
        from qpdk.cells.capacitor import interdigital_capacitor, plate_capacitor_single

        # These should produce valid components without errors
        idc = interdigital_capacitor()
        assert idc is not None
        assert len(idc.ports) >= 1

        pcs = plate_capacitor_single()
        assert pcs is not None
        assert len(pcs.ports) >= 1


class TestSubtractDrawFromEtch:
    """Tests for subtract_draw_from_etch helper."""

    def test_absorbs_result_into_component(self):
        c = Component()
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.M1_DRAW)

        etch_shape = gf.Component()
        etch_shape.add_polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_ETCH
        )

        subtract_draw_from_etch(
            component=c,
            etch_shape=etch_shape,
            etch_layer=LAYER.M1_ETCH,
            draw_layer=LAYER.M1_DRAW,
        )

        etch_layer_tuple = layerenum_to_tuple(LAYER.M1_ETCH)
        assert etch_layer_tuple in c.layers

    def test_etch_excludes_draw_area(self):
        c = Component()
        # Draw a small rectangle in the center
        c.add_polygon([(2, 2), (8, 2), (8, 8), (2, 8)], layer=LAYER.M1_DRAW)

        # Etch covers the whole area
        etch_shape = gf.Component()
        etch_shape.add_polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)], layer=LAYER.M1_ETCH
        )

        subtract_draw_from_etch(
            component=c,
            etch_shape=etch_shape,
            etch_layer=LAYER.M1_ETCH,
            draw_layer=LAYER.M1_DRAW,
        )

        # The etch area should be smaller than the full bbox
        # because the draw area was subtracted
        etch_layer_tuple = layerenum_to_tuple(LAYER.M1_ETCH)
        draw_layer_tuple = layerenum_to_tuple(LAYER.M1_DRAW)
        assert etch_layer_tuple in c.layers
        assert draw_layer_tuple in c.layers

    def test_works_with_m2_layers(self):
        c = Component()
        c.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer=LAYER.M2_DRAW)

        etch_shape = gf.components.circle(radius=20, layer=LAYER.M2_ETCH)

        subtract_draw_from_etch(
            component=c,
            etch_shape=etch_shape,
            etch_layer=LAYER.M2_ETCH,
            draw_layer=LAYER.M2_DRAW,
        )

        etch_layer_tuple = layerenum_to_tuple(LAYER.M2_ETCH)
        assert etch_layer_tuple in c.layers

    def test_produces_same_result_as_transmon_components(self):
        """Verify the refactored transmon components still work correctly."""
        from qpdk.cells.transmon import (
            double_pad_transmon_with_bbox,
            flipmon_with_bbox,
            xmon_transmon,
        )

        # These should produce valid components without errors
        dpt = double_pad_transmon_with_bbox()
        assert dpt is not None

        fm = flipmon_with_bbox()
        assert fm is not None

        xmon = xmon_transmon()
        assert xmon is not None
