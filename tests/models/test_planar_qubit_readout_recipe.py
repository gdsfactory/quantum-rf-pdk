"""Local coupons retain isolated electrodes and the actual coupling gap."""

import importlib
from itertools import product
from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk.cells import transmon_with_resonator_and_probeline
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


def test_nominal_transmon_matches_assembly_electrodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    recipe = importlib.import_module("double_pad_readout").RECIPE
    select_metal = importlib.import_module("_readout").select_metal
    point = recipe.provenance["nominal_layout"]
    assert all(point[name] in values for name, values in recipe.grid.items())
    geometry = recipe.geometry(point, recipe.settings, 4)
    metal = geometry.component.get_region(LAYER.M1_DRAW)
    local = gf.kdb.Region()
    for port in geometry.component.ports:
        local |= select_metal(metal, port, geometry.component.kcl.dbu)
    local.transform(gf.kdb.Trans(1, False, 0, 0))

    assembly = transmon_with_resonator_and_probeline()
    reference = gf.kdb.Region()
    for instance in assembly.insts:
        if instance.cell.function_name in {
            "double_pad_transmon_with_bbox",
            "plate_capacitor_single",
        }:
            reference |= gf.kdb.Region(
                instance.cell.begin_shapes_rec(gf.get_layer(LAYER.M1_DRAW))
            ).transformed(instance.cplx_trans)
    assert not reference.is_empty()
    assert (local ^ reference).is_empty()
