"""The bare bonded coupon keeps the QPDK electrodes without a readout paddle."""

import importlib
from itertools import product
from math import prod
from pathlib import Path
from typing import Any

import gdsfactory as gf
import pytest

from qpdk.cells import flipmon_with_bbox
from qpdk.models.datasets import Dataset
from qpdk.tech import LAYER


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes")
    )
    return importlib.import_module("flipmon_bonded")


def test_default_grid_matches_published_training_coordinates(recipe: Any) -> None:
    data = Dataset("flipmon_bonded_palace")
    grid = data.grid("maxwell_capacitance", bond="physical_bump", allow_missing=False)
    assert tuple(recipe.GRID) == tuple(axis.name for axis in data.metadata.axes)
    assert {
        axis.name: coords.tolist()
        for axis, coords in zip(grid.axes, grid.coords, strict=True)
    } == recipe.GRID
    assert prod(map(len, recipe.GRID.values())) == 189
    for axis in data.metadata.axes:
        assert axis.validated == (
            min(recipe.GRID[axis.name]),
            max(recipe.GRID[axis.name]),
        )


def test_entire_grid_preserves_both_metal_faces(recipe: Any) -> None:
    for values in product(*recipe.GRID.values()):
        point = dict(zip(recipe.GRID, values, strict=True))
        data = recipe.geometry(point, recipe.SETTINGS, 1)
        source = flipmon_with_bbox(
            inner_circle_radius=point["inner_radius"],
            top_circle_radius=point["top_radius"],
        )
        dbu = source.kcl.dbu
        frame = gf.kdb.Region(gf.kdb.DBox(-400, -400, 400, 400).to_itype(dbu))
        expected_bottom = (
            (frame - source.get_region(LAYER.M1_ETCH))
            | source.get_region(LAYER.M1_DRAW)
        ).merged()
        top = gf.Component()
        top.add_ref(source.extract([LAYER.M2_DRAW, LAYER.M2_ETCH])).dmovex(
            point["alignment"]
        )
        expected_top = (
            (frame - top.get_region(LAYER.M2_ETCH)) | top.get_region(LAYER.M2_DRAW)
        ).merged()
        for layer, expected in (
            (LAYER.M1_DRAW, expected_bottom),
            (LAYER.M2_DRAW, expected_top),
            (LAYER.IND, source.get_region(LAYER.IND)),
        ):
            actual = data.component.get_region(layer)
            assert (actual - expected.sized(1)).is_empty()
            assert (expected - actual.sized(1)).is_empty()
            assert (actual ^ expected).area() * dbu**2 < 1e-3
        assert data.component.get_region(LAYER.M1_DRAW).count() == 3
        assert data.component.get_region(LAYER.M2_DRAW).count() == 2
        assert data.terminal_ports == {"island": "island", "ring": "ring"}
        assert tuple(data.component.ports["island"].center) == (0.0, 0.0)
        assert tuple(data.component.ports["ring"].center) == (100.0, 0.0)
        stack = data.mesh_inputs["stack"]
        assert stack["layers"]["top"]["zmin"] == pytest.approx(point["chip_gap"] + 0.2)
        assert stack["layers"]["bump"]["thickness"] == pytest.approx(point["chip_gap"])
        for name in ("bottom_chip", "top_chip"):
            dielectric = next(d for d in stack["dielectrics"] if d["name"] == name)
            assert dielectric["zmax"] - dielectric["zmin"] == pytest.approx(300.0)


def test_ideal_wire_control_omits_only_the_bump(recipe: Any) -> None:
    point = dict(chip_gap=10.0, alignment=15.0, inner_radius=60.0, top_radius=100.0)
    physical = recipe.geometry(point, recipe.SETTINGS, 1)
    ideal = recipe.geometry(point | {"bond": "ideal_wire"}, recipe.SETTINGS, 1)
    assert not physical.component.get_region(LAYER.IND).is_empty()
    assert ideal.component.get_region(LAYER.IND).is_empty()
    for layer in (LAYER.M1_DRAW, LAYER.M2_DRAW):
        assert (
            physical.component.get_region(layer) ^ ideal.component.get_region(layer)
        ).is_empty()
    assert physical.mesh_inputs["bond"] == "physical_bump"
    assert ideal.mesh_inputs["bond"] == "ideal_wire"


@pytest.mark.parametrize(
    "change",
    [
        {"chip_gap": 0.0},
        {"inner_radius": 7.5},
        {"inner_radius": 80.0},
        {"alignment": 102.5},
        {"top_radius": 0.0},
        {"bond": "unknown"},
    ],
)
def test_invalid_geometry_is_rejected(
    recipe: Any, change: dict[str, float | str]
) -> None:
    point = dict(chip_gap=10.0, alignment=0.0, inner_radius=60.0, top_radius=110.0)
    with pytest.raises(
        ValueError, match=r"positive|inner_radius|land inside|Unknown bond"
    ):
        recipe.geometry(point | change, recipe.SETTINGS, 1)
