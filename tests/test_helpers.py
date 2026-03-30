"""Tests for qpdk.cells.helpers."""

import gdsfactory as gf
import pytest

from qpdk.cells.helpers import add_rect
from qpdk.tech import LAYER


def test_add_rect_x0_x1() -> None:
    """Test add_rect with x0, x1 coordinates."""
    c = gf.Component()
    add_rect(c, layer=LAYER.M1_DRAW, x0=0, x1=10, y0=0, y1=20)
    layer_index = c.kcl.layer(*LAYER.M1_DRAW)
    assert c.shapes(layer_index).size() == 1
    bbox = c.bbox()
    assert bbox.left == 0
    assert bbox.right == 10
    assert bbox.bottom == 0
    assert bbox.top == 20


def test_add_rect_center_width() -> None:
    """Test add_rect with x_center, width coordinates."""
    c = gf.Component()
    add_rect(c, layer=LAYER.M1_DRAW, x_center=5, width=10, y_center=10, height=20)
    layer_index = c.kcl.layer(*LAYER.M1_DRAW)
    assert c.shapes(layer_index).size() == 1
    bbox = c.bbox()
    assert bbox.left == 0
    assert bbox.right == 10
    assert bbox.bottom == 0
    assert bbox.top == 20


def test_add_rect_mixed() -> None:
    """Test add_rect with mixed coordinate types."""
    c = gf.Component()
    add_rect(c, layer=LAYER.M1_DRAW, x0=0, x1=10, y_center=10, height=20)
    layer_index = c.kcl.layer(*LAYER.M1_DRAW)
    assert c.shapes(layer_index).size() == 1
    bbox = c.bbox()
    assert bbox.left == 0
    assert bbox.right == 10
    assert bbox.bottom == 0
    assert bbox.top == 20


def test_add_rect_invalid_x() -> None:
    """Test add_rect with missing x coordinates."""
    c = gf.Component()
    with pytest.raises(ValueError, match=r"Provide \(x0, x1\) or \(x_center, width\)"):
        add_rect(c, layer=LAYER.M1_DRAW, x0=0, y0=0, y1=20)


def test_add_rect_invalid_y() -> None:
    """Test add_rect with missing y coordinates."""
    c = gf.Component()
    with pytest.raises(ValueError, match=r"Provide \(y0, y1\) or \(y_center, height\)"):
        add_rect(c, layer=LAYER.M1_DRAW, x0=0, x1=10, y0=0)
