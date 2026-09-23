"""Tests for the COMSOL geometry builder's public surface.

The builder talks to COMSOL through MPh, which is not installed here. These
tests cover only what runs before the client is touched (argument validation)
plus the compatibility alias, so nothing needs to mock MPh or the Java API.
"""

from __future__ import annotations

from typing import Any

import pytest

from qpdk import simulation
from qpdk.simulation import (
    build_comsol_cpw_model,
    build_comsol_metal_model,
    comsol_layout,
)
from qpdk.simulation.comsol import (
    build_comsol_cpw_model as comsol_cpw_model,
    build_comsol_metal_model as comsol_metal_model,
)
from qpdk.simulation.comsol_layout import ComsolBoundingBox, ComsolLayout


def _empty_layout() -> ComsolLayout:
    """A layout with a bounding box but no metal to extrude.

    Returns:
        The empty layout.
    """
    return ComsolLayout(
        polygons=(),
        feed_ports=(),
        bbox=ComsolBoundingBox(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0),
    )


def test_generic_and_cpw_names_are_the_same_builder():
    """The old CPW name is the generic builder, not a re-implementation."""
    assert build_comsol_cpw_model is build_comsol_metal_model
    assert comsol_cpw_model is comsol_metal_model
    assert build_comsol_metal_model.__name__ == "build_comsol_metal_model"


@pytest.mark.parametrize(
    "name",
    [
        "ComsolBoundingBox",
        "ComsolFeedPort",
        "ComsolLayout",
        "ComsolPolygon",
        "prepare_comsol_layout",
    ],
)
def test_comsol_layout_public_exports(name: str):
    """The lazy package exports resolve to the geometry module's objects."""
    assert getattr(simulation, name) is getattr(comsol_layout, name)


@pytest.mark.parametrize("thickness", [0.0, -0.2, float("nan"), float("inf")])
def test_rejects_bad_thickness_before_touching_comsol(thickness: float):
    """Thickness is validated before any MPh/COMSOL call."""
    client: Any = None
    with pytest.raises(ValueError, match="metal_thickness_um"):
        build_comsol_metal_model(client, _empty_layout(), metal_thickness_um=thickness)


def test_rejects_layout_without_polygons_before_touching_comsol():
    """An empty layout is refused before any MPh/COMSOL call."""
    client: Any = None
    with pytest.raises(ValueError, match="no polygons"):
        build_comsol_metal_model(client, _empty_layout())
