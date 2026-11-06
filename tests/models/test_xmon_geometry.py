"""Bare Xmon extraction preserves the cross without adding a readout conductor."""

import importlib
from itertools import product
from pathlib import Path
from typing import Any

import gdsfactory as gf
import pytest

from qpdk.cells import xmon_transmon
from qpdk.tech import LAYER


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes")
    )
    return importlib.import_module("xmon")


def test_complete_grid_preserves_the_island(recipe: Any) -> None:
    for scale, gap in product(*recipe.GRID.values()):
        result = recipe.geometry(
            {"arm_scale": scale, "gap_width": gap}, recipe.SETTINGS, 1
        )
        assert tuple(result.terminal_ports) == ("island",)
        assert {p.port_type for p in result.component.ports} == {"electrical"}
        assert result.component.layers == [gf.get_layer_tuple(LAYER.M1_DRAW)]
        source = xmon_transmon(
            arm_lengths=tuple(scale * length for length in (160, 120, 160, 120)),
            gap_width=gap,
        )
        metal = result.component.get_region(LAYER.M1_DRAW)
        pin = result.component.ports["island"]
        dbu = result.component.kcl.dbu
        x, y = round(pin.dx / dbu), round(pin.dy / dbu)
        island = metal.interacting(
            gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2))
        )
        assert (island ^ source.get_region(LAYER.M1_DRAW)).is_empty()
        assert metal.count() == 2
        ground = metal - island
        assert not ground.is_empty()
        assert (island & ground).is_empty()
        bounds = metal.bbox().to_dtype(dbu)
        pad = recipe.SETTINGS.domain_pad
        assert result.bounds == (
            bounds.left - pad,
            bounds.bottom - pad,
            bounds.right + pad,
            bounds.top + pad,
        )


@pytest.mark.parametrize(
    "point",
    [{"arm_scale": 0.0, "gap_width": 10.0}, {"arm_scale": 1.0, "gap_width": -1.0}],
)
def test_invalid_geometry_is_rejected(recipe, point: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        recipe.geometry(point, recipe.SETTINGS, 1)
