"""Checks for geometry passed from qpdk to MATLAB."""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.simulation.matlab import (
    component_to_terminal_polygons,
    interdigital_capacitor_fem_geometry,
)


def test_idc_fem_geometry_matches_layout_ports() -> None:
    PDK.activate()
    idc = interdigital_capacitor(
        fingers=4, finger_length=20.0, finger_gap=2.0, thickness=5.0, etch_layer=None
    )
    combs, bbox, thickness, eps_substrate, eps_vacuum = (
        interdigital_capacitor_fem_geometry(
            PDK, fingers=4, finger_length=20.0, finger_gap=2.0, finger_width=5.0
        )
    )

    assert len(combs) == len(idc.ports) == 2
    assert all(
        Polygon(xy).buffer(1e-3).covers(Point(*port.center))
        for xy, port in zip(combs, idc.ports, strict=True)
    )
    box = idc.dbbox()
    np.testing.assert_allclose(bbox, [box.left, box.bottom, box.right, box.top])
    assert thickness > 0
    assert eps_substrate > eps_vacuum == 1


def test_terminal_polygons_reject_missing_metal() -> None:
    PDK.activate()
    idc = interdigital_capacitor(fingers=4, etch_layer=None)
    with pytest.raises(ValueError, match="Every port must lie"):
        component_to_terminal_polygons(idc, layer=(999, 0))
