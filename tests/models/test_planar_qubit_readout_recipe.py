"""Local coupons retain isolated electrodes and the actual coupling gap."""

import importlib
from itertools import product
from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk.tech import LAYER


@pytest.mark.parametrize("name", ["double_pad_readout", "xmon_readout"])
def test_complete_geometry_grid(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module(name)
    recipe = module.RECIPE
    for values in product(*recipe.grid.values()):
        point = dict(zip(recipe.grid, values, strict=True))
        geometry = recipe.geometry(point, recipe.settings, 4)
        component = geometry.component
        dbu = component.kcl.dbu
        metal = component.get_region(LAYER.M1_DRAW)
        electrodes = []
        for terminal in recipe.terminals:
            port = component.ports[terminal]
            x, y = (round(value / dbu) for value in port.center)
            electrode = metal.interacting(
                gf.kdb.Region(gf.kdb.Box(x - 2, y - 2, x + 2, y + 2))
            )
            assert electrode.count() == 1
            electrodes.append(electrode)
        assert all(
            (a & b).is_empty() for i, a in enumerate(electrodes) for b in electrodes[:i]
        )
        assert metal.count() == len(recipe.terminals) + 1
        island = electrodes[0].bbox().to_dtype(dbu)
        paddle = electrodes[-1].bbox().to_dtype(dbu)
        assert island.left - paddle.right == pytest.approx(point["coupling_gap"])
        assert tuple(geometry.terminal_ports) == recipe.terminals
        assert component.get_region(LAYER.JJ_AREA).is_empty()
        assert component.get_region(LAYER.JJ_PATCH).is_empty()


@pytest.mark.parametrize("name", ["double_pad_readout", "xmon_readout"])
def test_invalid_geometry(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = importlib.import_module(name).RECIPE
    point = {key: values[1] for key, values in recipe.grid.items()}
    with pytest.raises(ValueError, match="positive"):
        recipe.geometry({**point, "coupling_gap": 0}, recipe.settings, 4)
